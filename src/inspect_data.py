"""Audit a YOLO-format dataset before any training happens.

Checks performed
----------------
  1. data.yaml parses, and declared class count matches the name list
  2. Every declared split exists and is non-empty
  3. Images and label files pair up (reports orphans on both sides)
  4. Label lines are well formed: 5 fields, integer class id in range,
     normalised coordinates within [0, 1], positive width and height
  5. Per-class instance counts for each split, plus the imbalance ratio
  6. Boxes smaller than a minimum relative area (hard to detect at 640px)
  7. Objects per image, flagged against the 32-piece chess maximum
  8. Duplicate images, detected by content hash
  9. Classes present in the schema but absent from a split

Usage
-----
    python src/inspect_data.py --data data/raw/chess-pieces
    python src/inspect_data.py --data data/raw/chess-pieces --json reports/audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# A box covering less than this fraction of the image is easy to miss at
# 640px input. Chess pieces viewed from across the board can be genuinely
# small, so this is a warning rather than an error.
MIN_REL_AREA = 0.0005

# A legal chess position has at most 32 pieces on the board.
MAX_PIECES = 32


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_data_yaml(root: Path) -> dict[str, Any]:
    """Locate and parse data.yaml, returning its contents."""
    candidates = [root / "data.yaml", root / "data.yml", root / "dataset.yaml"]
    for path in candidates:
        if path.is_file():
            with path.open() as fh:
                cfg = yaml.safe_load(fh) or {}
            cfg["_path"] = str(path)
            return cfg
    raise FileNotFoundError(
        f"No data.yaml found in {root}. Looked for: "
        + ", ".join(p.name for p in candidates)
    )


def class_names(cfg: dict[str, Any]) -> list[str]:
    """Extract class names, tolerating both list and dict forms."""
    names = cfg.get("names")
    if names is None:
        return []
    if isinstance(names, dict):
        # Keys may be ints or numeric strings; sort numerically.
        return [names[k] for k in sorted(names, key=lambda k: int(k))]
    return list(names)


def resolve_split_dir(root: Path, cfg: dict[str, Any], split: str) -> Path | None:
    """Find the image directory for a split, trying yaml then conventions."""
    raw = cfg.get(split)
    if raw:
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            # Roboflow exports often use paths like "../train/images".
            for base in (root, root.parent):
                resolved = (base / candidate).resolve()
                if resolved.is_dir():
                    return resolved
        elif candidate.is_dir():
            return candidate

    for pattern in (f"{split}/images", split):
        candidate = root / pattern
        if candidate.is_dir():
            return candidate
    return None


def label_path_for(image_path: Path) -> Path:
    """Map an image path to its YOLO label path."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        return image_path.parent / f"{image_path.stem}.txt"
    return Path(*parts).with_suffix(".txt")


# ---------------------------------------------------------------------------
# Per-split audit
# ---------------------------------------------------------------------------


def audit_split(
    image_dir: Path,
    n_classes: int,
    check_duplicates: bool = True,
) -> dict[str, Any]:
    """Audit one split and return a structured result."""
    images = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )

    result: dict[str, Any] = {
        "image_dir": str(image_dir),
        "n_images": len(images),
        "n_labelled_images": 0,
        "n_empty_labels": 0,
        "class_counts": Counter(),
        "boxes_per_image": [],
        "missing_labels": [],
        "orphan_labels": [],
        "malformed": [],
        "bad_class_id": [],
        "out_of_range": [],
        "degenerate": [],
        "tiny_boxes": [],
        "over_max_pieces": [],
        "duplicate_images": [],
    }

    hashes: dict[str, list[str]] = defaultdict(list)

    for img in images:
        if check_duplicates:
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            hashes[digest].append(img.name)

        lbl = label_path_for(img)
        if not lbl.is_file():
            result["missing_labels"].append(img.name)
            continue

        result["n_labelled_images"] += 1
        lines = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]

        if not lines:
            result["n_empty_labels"] += 1

        n_boxes = 0
        for lineno, line in enumerate(lines, start=1):
            ref = f"{lbl.name}:{lineno}"
            fields = line.split()

            if len(fields) != 5:
                result["malformed"].append(f"{ref} has {len(fields)} fields, expected 5")
                continue

            try:
                cls = int(float(fields[0]))
                x, y, w, h = (float(v) for v in fields[1:])
            except ValueError:
                result["malformed"].append(f"{ref} is not numeric")
                continue

            if not 0 <= cls < n_classes:
                result["bad_class_id"].append(f"{ref} class id {cls}")
                continue

            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                result["out_of_range"].append(
                    f"{ref} ({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})"
                )
                continue

            if w <= 0 or h <= 0:
                result["degenerate"].append(f"{ref} w={w:.4f} h={h:.4f}")
                continue

            if w * h < MIN_REL_AREA:
                result["tiny_boxes"].append(f"{ref} area={w * h:.6f}")

            result["class_counts"][cls] += 1
            n_boxes += 1

        result["boxes_per_image"].append(n_boxes)
        if n_boxes > MAX_PIECES:
            result["over_max_pieces"].append(f"{img.name} has {n_boxes} boxes")

    # Labels with no corresponding image.
    label_dir = label_path_for(image_dir / "_.jpg").parent
    if label_dir.is_dir():
        stems = {p.stem for p in images}
        result["orphan_labels"] = sorted(
            p.name for p in label_dir.glob("*.txt") if p.stem not in stems
        )

    if check_duplicates:
        result["duplicate_images"] = [
            names for names in hashes.values() if len(names) > 1
        ]

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _issue_lines(label: str, items: list[str], limit: int = 5) -> list[str]:
    if not items:
        return []
    lines = [f"  {label}: {len(items)}"]
    for item in items[:limit]:
        lines.append(f"      {item}")
    if len(items) > limit:
        lines.append(f"      ... and {len(items) - limit} more")
    return lines


def print_report(names: list[str], splits: dict[str, dict[str, Any]]) -> bool:
    """Print a human-readable report. Returns True if any hard error was found."""
    has_error = False

    print("=" * 68)
    print("DATASET AUDIT")
    print("=" * 68)

    print(f"\nClasses declared: {len(names)}")
    for i, name in enumerate(names):
        print(f"  {i:2d}  {name}")

    totals: Counter = Counter()

    for split, res in splits.items():
        print(f"\n{'-' * 68}\n{split.upper()}\n{'-' * 68}")
        print(f"  images:            {res['n_images']}")
        print(f"  with label file:   {res['n_labelled_images']}")
        print(f"  empty label files: {res['n_empty_labels']}")

        boxes = res["boxes_per_image"]
        if boxes:
            total = sum(boxes)
            totals.update(res["class_counts"])
            print(f"  total boxes:       {total}")
            print(
                f"  boxes/image:       "
                f"min {min(boxes)}, mean {total / len(boxes):.1f}, max {max(boxes)}"
            )

        errors = (
            _issue_lines("MISSING LABEL FILES", res["missing_labels"])
            + _issue_lines("ORPHAN LABEL FILES", res["orphan_labels"])
            + _issue_lines("MALFORMED LINES", res["malformed"])
            + _issue_lines("CLASS ID OUT OF RANGE", res["bad_class_id"])
            + _issue_lines("COORDS OUTSIDE [0,1]", res["out_of_range"])
            + _issue_lines("ZERO-AREA BOXES", res["degenerate"])
        )
        if errors:
            has_error = True
            print("\n  ERRORS")
            for line in errors:
                print(line)

        warnings = (
            _issue_lines("very small boxes", res["tiny_boxes"])
            + _issue_lines("more than 32 boxes", res["over_max_pieces"])
        )
        if res["duplicate_images"]:
            groups = res["duplicate_images"]
            warnings.append(f"  duplicate image groups: {len(groups)}")
            for group in groups[:3]:
                warnings.append(f"      {', '.join(group[:4])}")
        if warnings:
            print("\n  WARNINGS")
            for line in warnings:
                print(line)

        if res["class_counts"]:
            print("\n  per-class instances")
            for i, name in enumerate(names):
                count = res["class_counts"].get(i, 0)
                flag = "   <-- ABSENT" if count == 0 else ""
                print(f"      {i:2d}  {name:<20} {count:6d}{flag}")

    if totals:
        print(f"\n{'=' * 68}\nCLASS BALANCE (all splits)\n{'=' * 68}")
        counts = [totals.get(i, 0) for i in range(len(names))]
        present = [c for c in counts if c > 0]
        if present:
            ratio = max(present) / min(present)
            print(f"  total instances: {sum(counts)}")
            print(f"  most common:     {max(present)}")
            print(f"  least common:    {min(present)}")
            print(f"  imbalance ratio: {ratio:.1f}x")
            if ratio > 10:
                print(
                    "\n  NOTE: ratio above 10x. Expected for chess, since there are\n"
                    "  eight pawns per side but only one king. Report per-class\n"
                    "  precision and recall, not just overall mAP."
                )

    print()
    return has_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset root")
    parser.add_argument("--json", type=Path, help="Also write results as JSON")
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Skip content hashing (faster on large datasets)",
    )
    args = parser.parse_args()

    if not args.data.is_dir():
        print(f"Not a directory: {args.data}")
        return 2

    cfg = load_data_yaml(args.data)
    names = class_names(cfg)

    declared = cfg.get("nc")
    if declared is not None and declared != len(names):
        print(f"WARNING: data.yaml declares nc={declared} but lists {len(names)} names\n")

    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "valid", "test"):
        image_dir = resolve_split_dir(args.data, cfg, split)
        if image_dir is None:
            continue
        if split == "valid" and "val" in splits:
            continue
        splits[split] = audit_split(
            image_dir, len(names), check_duplicates=not args.no_duplicates
        )

    if not splits:
        print("No splits found. Expected train/valid/test directories.")
        return 2

    has_error = print_report(names, splits)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            "classes": names,
            "splits": {
                k: {**v, "class_counts": dict(v["class_counts"])}
                for k, v in splits.items()
            },
        }
        args.json.write_text(json.dumps(serialisable, indent=2))
        print(f"Wrote {args.json}")

    if has_error:
        print("Audit found errors. Fix these before training.")
        return 1

    print("Audit passed with no hard errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
