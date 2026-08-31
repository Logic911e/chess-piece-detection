"""Fine-tune YOLOv8 on the converted chess dataset.

Defaults to yolov8s for 50 epochs, which fits inside a single free Colab
session on 4,400 training images. Checkpoints are written every epoch, so an
interrupted run resumes rather than restarts.

Usage
-----
    python src/train_yolo.py
    python src/train_yolo.py --model yolov8m.pt --epochs 100
    python src/train_yolo.py --resume            # continue an interrupted run

On Colab, mount Drive first and pass --project so weights survive the session:

    python src/train_yolo.py --project /content/drive/MyDrive/AIML339/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import config


def resolve_device(requested: str) -> str:
    """Pick a compute device, preferring an actual accelerator."""
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    # Apple Silicon. Ultralytics supports mps, though it is far slower than a
    # CUDA GPU and is really only sensible for a smoke test.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=config.PROCESSED_DIR / "synth" / "data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--model", default=config.YOLO_MODEL,
                        help="Base checkpoint (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=config.YOLO_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.YOLO_BATCH,
                        help="Batch size, or -1 to auto-size to the GPU")
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, mps, or a CUDA index like 0")
    parser.add_argument("--project", type=Path, default=config.RUNS_DIR,
                        help="Where to write runs")
    parser.add_argument("--name", default=None,
                        help="Run name (default: model-epochs-timestamp)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience in epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the most recent interrupted run")
    parser.add_argument("--no-aug", action="store_true",
                        help="Disable augmentation, for an ablation baseline")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed.\n"
            "Run: pip install ultralytics"
        )

    if not args.data.is_file():
        sys.exit(
            f"No data.yaml at {args.data}\n"
            "Run src/synth_to_yolo.py first."
        )

    device = resolve_device(args.device)
    if device == "cpu":
        print(
            "[warn] No GPU detected. Training on CPU will take many hours.\n"
            "       Use Colab with a T4 runtime instead.\n"
        )

    if args.resume:
        last = sorted(
            args.project.glob("*/weights/last.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not last:
            sys.exit(f"No checkpoint to resume under {args.project}")
        ckpt = last[-1]
        print(f"[resume] {ckpt}")
        model = YOLO(str(ckpt))
        model.train(resume=True)
        return 0

    name = args.name or (
        f"{Path(args.model).stem}-e{args.epochs}-"
        f"{datetime.now():%Y%m%d-%H%M}"
    )

    aug = {} if args.no_aug else dict(config.YOLO_AUG)
    if args.no_aug:
        # Turn every augmentation off explicitly rather than relying on
        # Ultralytics' defaults, which are not all zero.
        aug = {
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "degrees": 0.0, "translate": 0.0, "scale": 0.0,
            "fliplr": 0.0, "flipud": 0.0, "mosaic": 0.0,
            "erasing": 0.0,
        }

    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)
    print(f"  data:    {args.data}")
    print(f"  model:   {args.model}")
    print(f"  epochs:  {args.epochs}")
    print(f"  batch:   {args.batch}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  device:  {device}")
    print(f"  aug:     {'disabled' if args.no_aug else 'enabled'}")
    print(f"  run:     {args.project / name}")
    print()

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(args.project),
        name=name,
        workers=args.workers,
        patience=args.patience,
        seed=config.SEED,
        deterministic=True,
        save=True,
        save_period=1,      # checkpoint every epoch so a drop costs one epoch
        plots=True,
        val=True,
        **aug,
    )

    run_dir = args.project / name
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "model": args.model,
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "device": device,
                "seed": config.SEED,
                "augmentation": aug,
            },
            indent=2,
        )
    )

    print(f"\nWeights: {run_dir / 'weights' / 'best.pt'}")
    print("\nNext:")
    print(f"    python src/evaluate.py --weights {run_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
