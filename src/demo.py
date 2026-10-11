"""Live chess piece detection demo.

Usage
-----
    python src/demo.py --model runs/finetune/ft-n1442-20260908-002129/weights/best.pt

Press q to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import cv2

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=None,
                   help="Path to YOLOv8 best.pt (omit to test webcam only)")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default 0)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold (default 0.25)")
    args = p.parse_args()

    # Load model if provided
    model = None
    if args.model is not None:
        from ultralytics import YOLO
        model = YOLO(str(args.model))
        print(f"Loaded model: {args.model}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if model is not None:
            results = model(frame, conf=args.conf, verbose=False)[0]
            frame = results.plot()

        cv2.imshow("Chess piece detection — q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
