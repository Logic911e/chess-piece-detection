"""Cut per-piece crops from a YOLO dataset into an ImageFolder classification set.

Produces the input for the ResNet-50 baseline: each labelled piece becomes one
image filed under its class name, so the classifier is handed the localisation
for free and only has to name the piece. That is the point of the baseline.
Comparing it against the detector isolates classification from localisation,
which is the variable finding F7 is about.

Layout written:

    dest/<split>/<class-name>/<source-stem>_<index>.jpg

plus `manifest.csv` recording every crop's source image and pixel box, so any
crop can be traced back and the conversion can be audited without opening the
images.

Two caveats this script has to handle, both documented rather than hidden.

Aspect ratio is signal. A rook is squat and a king is tall, and squashing every
crop into a square would throw that away. Crops are therefore letterboxed onto
a square canvas, preserving shape, rather than resized anisotropically.

Box convention differs between the two datasets. chesscog boxes are amodal,
computed from all mesh vertices including occluded ones, so a synthetic crop
contains whatever is standing in front of the piece. ChessReD boxes are tight
on the visible piece. A classifier trained on one and tested on the other is
therefore seeing a box-convention change on top of the domain change. Passing
`--shrink F` scales every box toward its centre by factor F, which approximates
a modal box on the synthetic side and lets the two effects be separated
experimentally instead of argued about.

Usage
-----
    python src/make_crops.py --src data/processed/synth --dest data/crops/synth
    python src/make_crops.py --src data/processed/chessred2k \\
                             --dest data/crops/chessred2k
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import config

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into (class, cx, cy, w, h) rows, all normalised."""
    rows = []
    if not path.is_file():
        return rows
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
        cls = int(parts[0])
        if not 0 <= cls < len(config.CANONICAL_CLASSES):
            raise ValueError(f"{path}:{lineno}: class {cls} out of range")
        rows.append((cls, *(float(v) for v in parts[1:])))
    return rows


def to_pixel_box(cx: float, cy: float, w: float, h: float,
                 iw: int, ih: int, pad: float, shrink: float
                 ) -> tuple[int, int, int, int]:
    """Normalised centre box -> integer pixel box, padded, shrunk and clipped.

    `shrink` scales the box toward its centre before padding is applied, so
    shrink=1.0 leaves the box alone and shrink=0.8 keeps the middle 80%.
    """
    w *= shrink
    h *= shrink
    half_w = w * (1.0 + pad) / 2.0
    half_h = h * (1.0 + pad) / 2.0
    left = int(round((cx - half_w) * iw))
    right = int(round((cx + half_w) * iw))
    top = int(round((cy - half_h) * ih))
    bottom = int(round((cy + half_h) * ih))
    return (max(0, left), max(0, top), min(iw, right), min(ih, bottom))


def letterbox(img, size: int):
    """Resize preserving aspect ratio onto a square canvas, padded with black."""
    from PIL import Image

    w, h = img.size
    scale = size / max(w, h)
    new = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    img = img.resize(new, Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - new[0]) // 2, (size - new[1]) // 2))
    return canvas


def process_split(src: Path, dest: Path, split: str, args) -> dict:
    from PIL import Image

    img_dir = src / split / "images"
    lbl_dir = src / split / "labels"
    if not img_dir.is_dir():
        return {}

    images = sorted(p for p in img_dir.iterdir() if p.suffix in IMAGE_SUFFIXES)
    if not images:
        return {}

    out_root = dest / split
    for name in config.CANONICAL_CLASSES:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    labelled = 0
    too_small = 0
    manifest = []

    for n, img_path in enumerate(images, 1):
        rows = read_label(lbl_dir / f"{img_path.stem}.txt")
        labelled += len(rows)
        if not rows:
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            iw, ih = im.size
            for idx, (cls, cx, cy, w, h) in enumerate(rows):
                box = to_pixel_box(cx, cy, w, h, iw, ih, args.pad, args.shrink)
                bw, bh = box[2] - box[0], box[3] - box[1]
                if min(bw, bh) < args.min_side:
                    too_small += 1
                    continue
                name = config.CANONICAL_CLASSES[cls]
                out = out_root / name / f"{img_path.stem}_{idx:03d}.jpg"
                letterbox(im.crop(box), args.size).save(
                    out, "JPEG", quality=args.quality)
                counts[name] += 1
                manifest.append((split, img_path.name, idx, name,
                                 box[0], box[1], box[2], box[3]))

        if n % 200 == 0 or n == len(images):
            print(f"  {split}: {n}/{len(images)} images, "
                  f"{sum(counts.values())} crops", flush=True)

    return {"counts": counts, "labelled": labelled,
            "too_small": too_small, "manifest": manifest,
            "images": len(images)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="YOLO dataset root holding train/val/test")
    p.add_argument("--dest", type=Path, required=True,
                   help="Where to write the ImageFolder crop dataset")
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    p.add_argument("--size", type=int, default=128,
                   help="Output square size in pixels (default 128)")
    p.add_argument("--pad", type=float, default=0.08,
                   help="Fractional margin added around each box (default 0.08)")
    p.add_argument("--shrink", type=float, default=1.0,
                   help="Scale boxes toward their centre before padding. "
                        "Use <1 to approximate modal boxes on amodal data.")
    p.add_argument("--min-side", type=int, default=10,
                   help="Skip boxes whose shorter side is below this, in pixels")
    p.add_argument("--quality", type=int, default=90)
    args = p.parse_args()

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        sys.exit("Pillow not installed: pip install pillow")

    if not args.src.is_dir():
        sys.exit(f"Source not found: {args.src}")
    if not 0 < args.shrink <= 1.0:
        sys.exit("--shrink must be in (0, 1]")

    print(f"source: {args.src}")
    print(f"dest:   {args.dest}")
    print(f"crops:  {args.size}px square, pad {args.pad}, shrink {args.shrink}\n")

    total: Counter[str] = Counter()
    all_manifest = []
    summary = []
    for split in args.splits:
        res = process_split(args.src, args.dest, split, args)
        if not res:
            print(f"  {split}: not present, skipped")
            continue
        total.update(res["counts"])
        all_manifest.extend(res["manifest"])
        summary.append((split, res["images"], sum(res["counts"].values()),
                        res["labelled"], res["too_small"]))

    if not summary:
        sys.exit("No splits processed")

    args.dest.mkdir(parents=True, exist_ok=True)
    with (args.dest / "manifest.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["split", "image", "index", "class",
                     "left", "top", "right", "bottom"])
        wr.writerows(all_manifest)

    print("\n=== crops written ===")
    print(f"{'split':<8}{'images':>8}{'crops':>9}{'labelled':>10}{'skipped':>9}")
    for split, imgs, crops, labelled, small in summary:
        print(f"{split:<8}{imgs:>8}{crops:>9}{labelled:>10}{small:>9}")

    print("\n=== class balance (all splits) ===")
    grand = sum(total.values())
    for name in config.CANONICAL_CLASSES:
        c = total[name]
        print(f"  {name:<14}{c:>8}   {c / grand * 100:>5.1f}%")
    print(f"  {'TOTAL':<14}{grand:>8}")

    # The king check again: every legal position has exactly one king per side,
    # so king crops should equal the image count unless boxes were dropped for
    # being too small. A mismatch larger than the skipped count is a real bug.
    imgs_total = sum(s[1] for s in summary)
    skipped = sum(s[4] for s in summary)
    print("\n=== king sanity check ===")
    ok = True
    for king in ("white-king", "black-king"):
        diff = imgs_total - total[king]
        flag = "ok" if 0 <= diff <= skipped else "MISMATCH"
        if flag != "ok":
            ok = False
        print(f"  {king:<14}{total[king]:>8}  vs {imgs_total} images  "
              f"(diff {diff}, skipped {skipped})  {flag}")

    print(f"\nWrote {args.dest / 'manifest.csv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
