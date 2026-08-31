"""Evaluate a trained YOLO model, broken down by render conditions.

An aggregate mAP says the model works. It does not say *where* it fails. The
chesscog renders record the camera angle and lighting mode used for every
image, so performance can be reported per condition instead, turning a single
number into a curve.

Rather than reimplementing mAP, this builds a small symlinked subset for each
stratum, writes a data.yaml pointing at it, and calls Ultralytics' own
validator. The arithmetic is therefore identical to the headline metric.

Usage
-----
    python src/evaluate.py --weights runs/<run>/weights/best.pt
    python src/evaluate.py --weights best.pt --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Stratification (no torch needed, so this is independently testable)
# ---------------------------------------------------------------------------


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def angle_label(angle: float, bins: Sequence[tuple[int, int]]) -> str | None:
    """Name the bin an angle falls into, using half-open [lo, hi) intervals."""
    for lo, hi in bins:
        if lo <= angle < hi:
            return f"angle_{lo}-{hi - 1}"
    return None


def assign_strata(
    rows: Iterable[dict[str, str]],
    split: str,
    bins: Sequence[tuple[int, int]] = config.ANGLE_BINS,
) -> dict[str, list[str]]:
    """Group image filenames into evaluation strata.

    Returns a mapping of stratum name to image filenames. Every image lands in
    "overall", plus one lighting stratum and one angle stratum where the
    metadata allows.
    """
    strata: dict[str, list[str]] = {"overall": []}

    for row in rows:
        if row.get("split") != split:
            continue
        name = row.get("image")
        if not name:
            continue

        strata["overall"].append(name)

        mode = (row.get("lighting_mode") or "").strip()
        if mode:
            strata.setdefault(f"lighting_{mode}", []).append(name)

        raw = (row.get("camera_angle") or "").strip()
        if raw:
            try:
                label = angle_label(float(raw), bins)
            except ValueError:
                label = None
            if label:
                strata.setdefault(label, []).append(name)

    return strata


def build_stratum(
    name: str,
    images: Sequence[str],
    src_images: Path,
    src_labels: Path,
    out_root: Path,
    class_names: Sequence[str],
) -> Path | None:
    """Materialise one stratum as a symlinked YOLO dataset. Returns its yaml."""
    if not images:
        return None

    stratum = out_root / name
    img_dir = stratum / "images"
    lbl_dir = stratum / "labels"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    linked = 0
    for filename in images:
        src_img = src_images / filename
        if not src_img.exists():
            continue
        src_lbl = src_labels / (Path(filename).stem + ".txt")
        if not src_lbl.exists():
            continue
        (img_dir / filename).symlink_to(src_img.resolve())
        (lbl_dir / src_lbl.name).symlink_to(src_lbl.resolve())
        linked += 1

    if linked == 0:
        shutil.rmtree(stratum, ignore_errors=True)
        return None

    yaml_path = stratum / "data.yaml"
    lines = [
        f"path: {stratum.resolve()}",
        "train: images",
        "val: images",
        f"nc: {len(class_names)}",
        "names:",
        *(f"  - {n}" for n in class_names),
    ]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def extract_metrics(results: Any) -> dict[str, float]:
    """Pull the headline numbers out of an Ultralytics results object."""
    box = getattr(results, "box", None)
    if box is None:
        return {}

    def scalar(attr: str) -> float:
        value = getattr(box, attr, None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "mAP50-95": scalar("map"),
        "mAP50": scalar("map50"),
        "precision": scalar("mp"),
        "recall": scalar("mr"),
    }


def per_class_map(results: Any, class_names: Sequence[str]) -> dict[str, float]:
    box = getattr(results, "box", None)
    maps = getattr(box, "maps", None) if box is not None else None
    if maps is None:
        return {}
    out: dict[str, float] = {}
    for i, name in enumerate(class_names):
        try:
            out[name] = float(maps[i])
        except (IndexError, TypeError, ValueError):
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        default=config.PROCESSED_DIR / "synth",
                        help="Converted dataset root")
    parser.add_argument("--split", default="test",
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=config.REPORTS_DIR,
                        help="Where to write the results table")
    parser.add_argument("--keep-strata", action="store_true",
                        help="Do not delete the temporary stratum directories")
    args = parser.parse_args()

    if not args.weights.is_file():
        sys.exit(f"No weights at {args.weights}")

    meta_csv = args.data / "render_metadata.csv"
    if not meta_csv.is_file():
        sys.exit(
            f"No render_metadata.csv at {meta_csv}\n"
            "Re-run src/synth_to_yolo.py, which writes it."
        )

    src_images = args.data / args.split / "images"
    src_labels = args.data / args.split / "labels"
    if not src_images.is_dir():
        sys.exit(f"No images at {src_images}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed.\nRun: pip install ultralytics")

    from train_yolo import resolve_device

    device = resolve_device(args.device)
    class_names = list(config.CANONICAL_CLASSES)

    rows = load_metadata(meta_csv)
    strata = assign_strata(rows, args.split)
    if not strata.get("overall"):
        sys.exit(f"No metadata rows for split '{args.split}'")

    print("=" * 66)
    print(f"STRATIFIED EVALUATION  ({args.split} split)")
    print("=" * 66)
    for name in sorted(strata):
        print(f"  {name:24} {len(strata[name]):5d} images")
    print()

    work = args.data / "_strata"
    work.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    results_table: dict[str, dict[str, float]] = {}
    per_class: dict[str, dict[str, float]] = {}

    for name in sorted(strata, key=lambda n: (n != "overall", n)):
        yaml_path = build_stratum(
            name, strata[name], src_images, src_labels, work, class_names
        )
        if yaml_path is None:
            print(f"[skip] {name}: no usable images")
            continue

        print(f"\n[eval] {name} ({len(strata[name])} images)")
        res = model.val(
            data=str(yaml_path),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            verbose=False,
            plots=False,
        )
        results_table[name] = extract_metrics(res)
        per_class[name] = per_class_map(res, class_names)

    if not results_table:
        print("Nothing evaluated.")
        return 1

    # ---- report -----------------------------------------------------------
    print(f"\n{'=' * 66}\nRESULTS BY CONDITION\n{'=' * 66}")
    header = f"{'stratum':24} {'mAP50-95':>10} {'mAP50':>8} {'prec':>8} {'recall':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results_table.items():
        print(
            f"{name:24} {m.get('mAP50-95', float('nan')):10.4f}"
            f" {m.get('mAP50', float('nan')):8.4f}"
            f" {m.get('precision', float('nan')):8.4f}"
            f" {m.get('recall', float('nan')):8.4f}"
        )

    overall = per_class.get("overall", {})
    if overall:
        print(f"\n{'=' * 66}\nPER-CLASS mAP50-95 (overall)\n{'=' * 66}")
        for name, value in sorted(overall.items(), key=lambda kv: kv[1]):
            print(f"  {name:20} {value:.4f}")
        worst = min(overall, key=overall.get)
        best = max(overall, key=overall.get)
        print(f"\n  weakest: {worst} ({overall[worst]:.4f})")
        print(f"  best:    {best} ({overall[best]:.4f})")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(args.weights),
        "split": args.split,
        "counts": {k: len(v) for k, v in strata.items()},
        "metrics": results_table,
        "per_class_map": per_class,
    }
    json_path = args.out / f"eval_{args.split}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = args.out / f"eval_{args.split}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stratum", "n_images", "mAP50-95", "mAP50", "precision", "recall"])
        for name, m in results_table.items():
            writer.writerow([
                name, len(strata.get(name, [])),
                f"{m.get('mAP50-95', ''):.4f}", f"{m.get('mAP50', ''):.4f}",
                f"{m.get('precision', ''):.4f}", f"{m.get('recall', ''):.4f}",
            ])

    if not args.keep_strata:
        shutil.rmtree(work, ignore_errors=True)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
