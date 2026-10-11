"""Central configuration for the chess piece detection project.

Paths are resolved relative to the repository root so the same code runs
locally and on Colab without editing.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/config.py -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # datasets exactly as downloaded
INTERIM_DIR = DATA_DIR / "interim"  # after class remapping / merging
PROCESSED_DIR = DATA_DIR / "processed"  # final YOLO-ready dataset
REPORTS_DIR = REPO_ROOT / "reports"
RUNS_DIR = REPO_ROOT / "runs"

# On Colab, persist heavy artefacts to Drive instead of the ephemeral VM disk.
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/AIML339/chess-detection")


def on_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    return "COLAB_GPU" in os.environ or Path("/content").is_dir()


def ensure_dirs() -> None:
    """Create every project directory that should exist."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Class schema
# ---------------------------------------------------------------------------

# The canonical 12 classes: six piece types in each of two colours.
#
# NOTE: the design report currently states 13 classes while describing 12.
# Six piece types x two colours is 12. Some public chess datasets add a
# thirteenth "board" class; whether this dataset does is checked empirically
# by inspect_data.py rather than assumed here.
PIECE_TYPES = ("pawn", "knight", "bishop", "rook", "queen", "king")
COLOURS = ("white", "black")

CANONICAL_CLASSES = tuple(
    f"{colour}-{piece}" for colour in COLOURS for piece in PIECE_TYPES
)
assert len(CANONICAL_CLASSES) == 12

# Class names vary between sources ("white-pawn", "P", "WhitePawn", ...).
# This maps the raw names onto CANONICAL_CLASSES. It is deliberately empty
# until inspect_data.py reports what the downloaded data actually contains.
CLASS_ALIASES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

SEED = 42
IMG_SIZE = 640

# yolov8s rather than yolov8m: roughly 3x faster, and 50 epochs on 4,400
# images fits inside a single free Colab session instead of being reclaimed
# partway through.
YOLO_MODEL = "yolov8s.pt"
YOLO_EPOCHS = 50
YOLO_BATCH = 16

# Augmentation. Defaults are Ultralytics' own unless noted.
#
# fliplr is safe here: mirroring a board leaves every piece identity intact.
# flipud is left at 0 deliberately, since an upside-down board never occurs in
# reality and training on one would only waste capacity.
YOLO_AUG = {
    "hsv_h": 0.015,   # hue jitter
    "hsv_s": 0.7,     # saturation jitter
    "hsv_v": 0.4,     # brightness jitter, stands in for the report's
                      # brightness/contrast augmentation
    "degrees": 15.0,  # rotation, matching the report's +/-15 degrees
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,    # disabled for the final epochs via close_mosaic
    "close_mosaic": 10,
}

# Camera angle buckets for stratified evaluation. The renderer sampled
# uniformly between 45 and 60 degrees.
ANGLE_BINS = ((45, 50), (50, 55), (55, 61))

RESNET_EPOCHS_HEAD = 5     # head only, backbone frozen
RESNET_EPOCHS_FULL = 10    # full network, reduced learning rate
RESNET_LR_HEAD = 1e-3
RESNET_LR_FULL = 1e-4
