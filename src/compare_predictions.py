"""Render side-by-side predictions from two checkpoints on the same photograph.

Produces the single most direct picture of the domain gap: the synthetic-trained
model's output next to the fine-tuned model's output on an identical real image.

Usage
-----
    python src/compare_predictions.py \\
        --before runs/best.pt \\
        --after "runs/finetune/ft-n1442-.../weights/best.pt" \\
        --n 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, default=Path("runs/best.pt"),
                   help="Synthetic-trained checkpoint")
    p.add_argument("--after", type=Path, default=None,
                   help="Fine-tuned checkpoint (default: newest finetune run)")
    p.add_argument("--data", type=Path,
                   default=config.PROCESSED_DIR / "chessred2k")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=2, help="How many photographs")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path,
                   default=Path("../prediction_comparison"))
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed")
    import cv2
    import numpy as np

    after = args.after
    if after is None:
        cands = sorted(Path("runs/finetune").glob("*/weights/best.pt"),
                       key=lambda q: q.stat().st_mtime)
        if not cands:
            sys.exit("No fine-tuned checkpoint found under runs/finetune")
        after = cands[-1]

    print(f"before: {args.before}")
    print(f"after:  {after}\n")

    imgs = sorted((args.data / args.split / "images").iterdir())[: args.n * 7 : 7]
    imgs = imgs[: args.n]
    if not imgs:
        sys.exit("No images found")

    m_before, m_after = YOLO(str(args.before)), YOLO(str(after))
    args.out.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        panels = []
        for model, tag in ((m_before, "SYNTHETIC ONLY"), (m_after, "FINE-TUNED")):
            r = model.predict(str(img), conf=args.conf, verbose=False)[0]
            frame = r.plot(line_width=4, font_size=12)
            h, w = frame.shape[:2]
            band = np.zeros((90, w, 3), dtype=np.uint8)
            n_det = len(r.boxes)
            cv2.putText(band, f"{tag}   ({n_det} detections)", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
            panels.append(np.vstack([band, frame]))

        h = min(p.shape[0] for p in panels)
        panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
                  for p in panels]
        combined = np.hstack(panels)
        scale = min(1.0, 2400 / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        dest = args.out / f"compare_{img.stem}.jpg"
        cv2.imwrite(str(dest), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
