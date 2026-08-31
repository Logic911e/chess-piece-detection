"""Convert the chesscog rendered dataset into YOLO detection format.

The rendered dataset from Woelflein and Arandjelovic ships one JSON per image:

    {
      "fen": "1KR2R2/P1PPQ2P/...",
      "white_turn": true,
      "camera":   {"angle": 52, "location": [...]},
      "lighting": {"mode": "flash", ...},
      "corners":  [[x, y], ...],            # four board corners
      "pieces": [
        {"piece": "P", "square": "e2", "box": [x, y, w, h]},
        ...
      ]
    }

The per-piece "box" field is produced in Blender by projecting the piece's
mesh vertices through the camera, so the boxes are exact rather than
estimated. They are given in pixels as (left, top, width, height).

Piece symbols follow FEN convention: uppercase is white, lowercase is black,
which gives the 12 colour-separated classes this project needs.

Note on occlusion: boxes are computed from all mesh vertices, including those
hidden behind other pieces, so they are amodal. A piece partly behind another
gets a box covering its full extent rather than only its visible part. This is
reasonable for chess, but it is a real difference from hand-annotated datasets
and is worth stating in the report.

Usage
-----
    python src/synth_to_yolo.py --src data/raw/render --dest data/processed/synth
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

import config

# FEN symbol -> index into config.CANONICAL_CLASSES.
# CANONICAL_CLASSES is ordered colour-major: white pawn..king, then black.
SYMBOL_TO_INDEX: dict[str, int] = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,   # white
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,  # black
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class ConversionStats:
    """Counters describing what happened during a conversion run."""

    def __init__(self) -> None:
        self.images_converted = 0
        self.images_skipped_no_json = 0
        self.images_skipped_no_boxes = 0
        self.boxes_written = 0
        self.boxes_dropped_degenerate = 0
        self.boxes_dropped_unknown_symbol = 0
        self.boxes_clamped = 0
        self.class_counts: Counter = Counter()
        self.unknown_symbols: Counter = Counter()

    def summary(self) -> str:
        lines = [
            f"  images converted:        {self.images_converted}",
            f"  images with no JSON:     {self.images_skipped_no_json}",
            f"  images with no boxes:    {self.images_skipped_no_boxes}",
            f"  boxes written:           {self.boxes_written}",
            f"  boxes clamped to bounds: {self.boxes_clamped}",
            f"  boxes dropped (zero area): {self.boxes_dropped_degenerate}",
            f"  boxes dropped (bad symbol): {self.boxes_dropped_unknown_symbol}",
        ]
        if self.unknown_symbols:
            lines.append(f"  unknown symbols: {dict(self.unknown_symbols)}")
        return "\n".join(lines)


def find_images(src: Path) -> Iterator[Path]:
    """Yield rendered image paths, sorted for reproducibility."""
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def convert_boxes(
    pieces: list[dict[str, Any]],
    img_w: int,
    img_h: int,
    stats: ConversionStats,
) -> list[str]:
    """Turn chesscog piece records into YOLO label lines."""
    lines: list[str] = []

    for piece in pieces:
        symbol = piece.get("piece")
        box = piece.get("box")

        if symbol not in SYMBOL_TO_INDEX:
            stats.boxes_dropped_unknown_symbol += 1
            stats.unknown_symbols[str(symbol)] += 1
            continue

        if not box or len(box) != 4:
            stats.boxes_dropped_degenerate += 1
            continue

        left, top, width, height = (float(v) for v in box)

        # Convert corner+size in pixels to normalised centre form.
        cx = (left + width / 2.0) / img_w
        cy = (top + height / 2.0) / img_h
        nw = width / img_w
        nh = height / img_h

        clamped = False
        for value in (cx, cy, nw, nh):
            if not 0.0 <= value <= 1.0:
                clamped = True
        if clamped:
            stats.boxes_clamped += 1

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)

        if nw <= 0.0 or nh <= 0.0:
            stats.boxes_dropped_degenerate += 1
            continue

        cls = SYMBOL_TO_INDEX[symbol]
        stats.class_counts[cls] += 1
        stats.boxes_written += 1
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def convert_split(
    src: Path,
    dest: Path,
    split: str,
    copy_images: bool,
    metadata_rows: list[dict[str, Any]],
) -> ConversionStats:
    """Convert one split and return its statistics."""
    stats = ConversionStats()

    img_out = dest / split / "images"
    lbl_out = dest / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path in find_images(src):
        json_path = img_path.with_suffix(".json")
        if not json_path.is_file():
            stats.images_skipped_no_json += 1
            continue

        with json_path.open() as fh:
            label = json.load(fh)

        pieces = label.get("pieces")
        if not pieces:
            # Older renders may predate the per-piece box field.
            stats.images_skipped_no_boxes += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        lines = convert_boxes(pieces, img_w, img_h, stats)
        if not lines:
            stats.images_skipped_no_boxes += 1
            continue

        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")

        target = img_out / img_path.name
        if copy_images:
            shutil.copy2(img_path, target)
        elif not target.exists():
            target.symlink_to(img_path.resolve())

        stats.images_converted += 1

        # Keep the render conditions so results can be broken down by
        # camera angle and lighting mode later.
        camera = label.get("camera") or {}
        lighting = label.get("lighting") or {}
        metadata_rows.append(
            {
                "image": img_path.name,
                "split": split,
                "camera_angle": camera.get("angle"),
                "lighting_mode": lighting.get("mode"),
                "white_turn": label.get("white_turn"),
                "n_pieces": len(lines),
            }
        )

    return stats


def write_data_yaml(dest: Path, splits: list[str]) -> Path:
    """Write the Ultralytics data.yaml for the converted dataset."""
    names = list(config.CANONICAL_CLASSES)
    lines = []
    for split in splits:
        key = "val" if split in ("val", "valid") else split
        lines.append(f"{key}: {split}/images")
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    lines.extend(f"  - {name}" for name in names)

    path = dest / "data.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_metadata_csv(dest: Path, rows: list[dict[str, Any]]) -> Path | None:
    """Write render conditions to CSV for stratified evaluation."""
    if not rows:
        return None
    import csv

    path = dest / "render_metadata.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Root of the downloaded render dataset",
    )
    parser.add_argument(
        "--dest", type=Path, default=config.PROCESSED_DIR / "synth",
        help="Where to write the YOLO dataset",
    )
    parser.add_argument(
        "--splits", nargs="*", default=None,
        help="Split subdirectories to convert (default: auto-detect)",
    )
    parser.add_argument(
        "--copy-images", action="store_true",
        help="Copy images instead of symlinking (needed on Colab/Drive)",
    )
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"Not a directory: {args.src}")
        return 2

    # Auto-detect splits, falling back to treating src as a single flat set.
    if args.splits is not None:
        splits = args.splits
    else:
        splits = [
            d.name for d in sorted(args.src.iterdir())
            if d.is_dir() and d.name in ("train", "val", "valid", "test")
        ]
    flat = not splits
    if flat:
        splits = ["train"]

    args.dest.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[dict[str, Any]] = []
    totals = Counter()
    all_classes: Counter = Counter()

    print("=" * 60)
    print("CHESSCOG RENDER -> YOLO")
    print("=" * 60)

    for split in splits:
        split_src = args.src if flat else args.src / split
        if not split_src.is_dir():
            print(f"\n[{split}] missing, skipped")
            continue

        print(f"\n[{split}]")
        stats = convert_split(
            split_src, args.dest, split, args.copy_images, metadata_rows
        )
        print(stats.summary())
        totals["images"] += stats.images_converted
        totals["boxes"] += stats.boxes_written
        totals["no_boxes"] += stats.images_skipped_no_boxes
        all_classes.update(stats.class_counts)

    if totals["images"] == 0:
        print(
            "\nNo images converted.\n"
            "If every image was skipped for missing boxes, the released\n"
            "renders predate the per-piece box field and bounding boxes must\n"
            "be derived from 'corners' and 'fen' instead."
        )
        return 1

    yaml_path = write_data_yaml(args.dest, splits)
    csv_path = write_metadata_csv(args.dest, metadata_rows)

    print(f"\n{'=' * 60}\nTOTALS\n{'=' * 60}")
    print(f"  images: {totals['images']}")
    print(f"  boxes:  {totals['boxes']}")
    if totals["images"]:
        print(f"  mean boxes/image: {totals['boxes'] / totals['images']:.1f}")

    print("\n  per-class instances")
    for i, name in enumerate(config.CANONICAL_CLASSES):
        count = all_classes.get(i, 0)
        flag = "   <-- ABSENT" if count == 0 else ""
        print(f"      {i:2d}  {name:<16} {count:7d}{flag}")

    present = [c for c in all_classes.values() if c > 0]
    if present:
        print(f"\n  imbalance ratio: {max(present) / min(present):.1f}x")

    print(f"\nWrote {yaml_path}")
    if csv_path:
        print(f"Wrote {csv_path}")
    print(f"\nNext: python src/inspect_data.py --data {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
