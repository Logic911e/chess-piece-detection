"""Convert ChessReD2K annotations into YOLO detection format.

ChessReD ships a single annotations.json describing all 10,800 images, but only
the 2,078-image ChessReD2K subset carries bounding boxes. This converts that
subset, using ChessReD's own train/val/test split.

Two details make a naive conversion wrong:

1. **Category order differs from ours.** ChessReD orders pieces
   pawn, rook, knight, bishop, queen, king; this project uses
   pawn, knight, bishop, rook, queen, king. Mapping by index would relabel
   every rook as a knight and every knight as a bishop, silently, while still
   producing plausible-looking metrics. Classes are therefore matched **by
   name**, and the script aborts if any name fails to match.

2. **There is a thirteenth category, `empty`.** It carries no boxes and is not
   a piece, so it is excluded rather than mapped.

Boxes are COCO style: [x_left, y_top, width, height] in pixels, floats.
Image dimensions come from each image record rather than being assumed, since
the three handsets produced 3072x3072 and 3024x3024.

Usage
-----
    python src/chessred_to_yolo.py \\
        --src data/raw/chessred \\
        --dest data/processed/chessred2k
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import config

EXCLUDED_CATEGORIES = {"empty"}


def build_category_map(categories: list[dict[str, Any]]) -> dict[int, int]:
    """Map ChessReD category ids onto config.CANONICAL_CLASSES indices, by name.

    Aborts rather than guessing if a name cannot be matched, since a silent
    mismatch here corrupts every label in the dataset.
    """
    canonical = {name: i for i, name in enumerate(config.CANONICAL_CLASSES)}
    mapping: dict[int, int] = {}
    unmatched: list[str] = []

    for cat in categories:
        name = str(cat.get("name", "")).strip().lower()
        cid = cat.get("id")
        if name in EXCLUDED_CATEGORIES:
            continue
        if name in canonical:
            mapping[cid] = canonical[name]
        else:
            unmatched.append(f"{cid}:{name}")

    if unmatched:
        sys.exit(
            "Category names could not be matched to the project schema:\n"
            f"  unmatched: {unmatched}\n"
            f"  expected:  {list(config.CANONICAL_CLASSES)}\n"
            "Refusing to guess; a wrong mapping would corrupt every label."
        )

    missing = set(canonical) - {
        config.CANONICAL_CLASSES[v] for v in mapping.values()
    }
    if missing:
        sys.exit(f"No ChessReD category maps to: {sorted(missing)}")

    return mapping


def extract_split_ids(splits: dict[str, Any]) -> dict[str, set[int]]:
    """Pull the ChessReD2K image ids for each split, tolerating shape variation."""
    subset = splits.get("chessred2k")
    if not isinstance(subset, dict):
        sys.exit("No 'chessred2k' entry in splits; cannot locate the box subset.")

    out: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        entry = subset.get(split)
        if entry is None:
            continue
        if isinstance(entry, dict):
            ids = entry.get("image_ids")
        elif isinstance(entry, list):
            ids = entry
        else:
            ids = None
        if ids:
            out[split] = {int(i) for i in ids}

    if not out:
        sys.exit("Found 'chessred2k' but no image ids inside it.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=config.RAW_DIR / "chessred",
                        help="ChessReD root holding annotations.json and images/")
    parser.add_argument("--dest", type=Path,
                        default=config.PROCESSED_DIR / "chessred2k")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Continue if image files are absent (labels only)")
    args = parser.parse_args()

    ann_path = args.src / "annotations.json"
    if not ann_path.is_file():
        sys.exit(f"No annotations.json at {ann_path}\n"
                 "Run src/download_chessred.py first.")

    print("=" * 66)
    print("ChessReD2K -> YOLO")
    print("=" * 66)

    with ann_path.open() as fh:
        data = json.load(fh)

    cat_map = build_category_map(data["categories"])
    print("\ncategory mapping (ChessReD id -> our index)")
    for src_id in sorted(cat_map):
        name = config.CANONICAL_CLASSES[cat_map[src_id]]
        flag = "" if src_id == cat_map[src_id] else "   <-- reordered"
        print(f"  {src_id:>3} -> {cat_map[src_id]:>3}  {name}{flag}")

    split_ids = extract_split_ids(data["splits"])
    print("\nChessReD2K splits")
    for split, ids in split_ids.items():
        print(f"  {split:6} {len(ids):5d} images")

    images = {img["id"]: img for img in data["images"]}

    # annotations.json records paths for the full archive. The ChessReD2K
    # archive may lay its files out differently, so build a filename index to
    # fall back on rather than silently dropping every image.
    file_index: dict[str, Path] = {}
    for found in args.src.rglob("*.jpg"):
        file_index.setdefault(found.name, found)
    if file_index:
        print(f"\nindexed {len(file_index)} image files under {args.src}")

    # Group boxed annotations by image.
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for piece in data["annotations"]["pieces"]:
        if piece.get("bbox") is not None:
            by_image[piece["image_id"]].append(piece)
    print(f"\nimages carrying boxes: {len(by_image)}")

    args.dest.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    totals = Counter()
    class_counts: Counter = Counter()
    missing_files = 0
    dropped = Counter()

    for split, ids in split_ids.items():
        img_dir = args.dest / split / "images"
        lbl_dir = args.dest / split / "labels"
        for d in (img_dir, lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

        converted = 0
        for image_id in sorted(ids):
            pieces = by_image.get(image_id)
            if not pieces:
                continue
            img = images.get(image_id)
            if img is None:
                continue

            width = float(img.get("width") or 0)
            height = float(img.get("height") or 0)
            if width <= 0 or height <= 0:
                dropped["no_dimensions"] += 1
                continue

            lines = []
            for piece in pieces:
                cid = piece.get("category_id")
                if cid not in cat_map:
                    dropped["unmapped_category"] += 1
                    continue
                box = piece["bbox"]
                if not box or len(box) != 4:
                    dropped["malformed_box"] += 1
                    continue

                left, top, bw, bh = (float(v) for v in box)
                cx = (left + bw / 2.0) / width
                cy = (top + bh / 2.0) / height
                nw = bw / width
                nh = bh / height

                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
                if nw <= 0 or nh <= 0:
                    dropped["degenerate"] += 1
                    continue

                cls = cat_map[cid]
                class_counts[cls] += 1
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if not lines:
                continue

            name = str(img["file_name"])
            src_img = args.src / str(img["path"])
            if not src_img.is_file():
                # Fall back to locating the file by name anywhere under src.
                src_img = file_index.get(name, src_img)
            if not src_img.is_file():
                missing_files += 1
                if not args.allow_missing:
                    continue

            (lbl_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n")

            if src_img.is_file():
                target = img_dir / name
                if args.copy_images:
                    shutil.copy2(src_img, target)
                elif not target.exists():
                    target.symlink_to(src_img.resolve())

            metadata.append({
                "image": name,
                "split": split,
                "camera_angle": "",
                "lighting_mode": "",
                "camera": img.get("camera", ""),
                "game_id": img.get("game_id", ""),
                "move_id": img.get("move_id", ""),
                "n_pieces": len(lines),
            })
            converted += 1
            totals["boxes"] += len(lines)

        totals["images"] += converted
        print(f"  [{split}] converted {converted} images")

    if totals["images"] == 0:
        print("\nNothing converted.")
        return 1

    # data.yaml
    names = list(config.CANONICAL_CLASSES)
    yaml_lines = []
    for split in split_ids:
        key = "val" if split in ("val", "valid") else split
        yaml_lines.append(f"{key}: {split}/images")
    yaml_lines.append(f"nc: {len(names)}")
    yaml_lines.append("names:")
    yaml_lines.extend(f"  - {n}" for n in names)
    (args.dest / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    if metadata:
        with (args.dest / "render_metadata.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metadata[0]))
            writer.writeheader()
            writer.writerows(metadata)

    print(f"\n{'=' * 66}\nTOTALS\n{'=' * 66}")
    print(f"  images: {totals['images']}")
    print(f"  boxes:  {totals['boxes']}")
    print(f"  mean boxes/image: {totals['boxes'] / totals['images']:.1f}")
    if dropped:
        print(f"  dropped: {dict(dropped)}")
    if missing_files:
        print(f"  image files not found: {missing_files}"
              f"{' (labels written anyway)' if args.allow_missing else ''}")

    print("\n  per-class instances")
    for i, name in enumerate(names):
        count = class_counts.get(i, 0)
        flag = "   <-- ABSENT" if count == 0 else ""
        print(f"      {i:2d}  {name:<16} {count:7d}{flag}")

    present = [c for c in class_counts.values() if c > 0]
    if present:
        print(f"\n  imbalance ratio: {max(present) / min(present):.1f}x")

    cams = Counter(m["camera"] for m in metadata if m["camera"])
    if cams:
        print("\n  images per camera")
        for cam, n in cams.most_common():
            print(f"      {cam:24} {n:5d}")

    print(f"\nWrote {args.dest / 'data.yaml'}")
    print(f"\nNext: python src/inspect_data.py --data {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
