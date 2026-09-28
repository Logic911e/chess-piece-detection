"""Fine-tune the synthetic-trained detector on N real photographs.

The point is not a single fine-tuned model but a **data-efficiency curve**.
Wolflein and Arandjelovic reported that fine-tuning on just two real photos
took their pipeline from 9.33 wrong squares per board to 0.11. Running this
across N = 0, 2, 5, 10, 25, 50, 100, 250, 500, all tests whether the same
holds for an end-to-end detector, and says how many real images are actually
needed to close the domain gap.

N = 0 evaluates the synthetic model unchanged, giving the zero-shot baseline.

Sampling is stratified across games rather than uniform. ChessReD2K photographs
each game move by move, so consecutive images differ by a single move; drawing
N images at random would often return near-duplicates of one position. Spreading
across games maximises diversity for a given N, which is also how someone would
realistically collect a handful of photos.

Usage
-----
    python src/finetune.py --n-images 0            # zero-shot baseline
    python src/finetune.py --n-images 10
    python src/finetune.py --sweep                 # the full curve

Results append to reports/finetune_sweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import config

DEFAULT_SWEEP = (0, 2, 5, 10, 25, 50, 100, 250, 500, -1)


# ---------------------------------------------------------------------------
# Subset construction (torch not required, so independently testable)
# ---------------------------------------------------------------------------


def stems_by_game(meta_rows: Sequence[dict[str, str]], split: str) -> dict[str, list[str]]:
    """Group image stems by game id for one split."""
    games: dict[str, list[str]] = defaultdict(list)
    for row in meta_rows:
        if row.get("split") != split:
            continue
        name = row.get("image")
        if not name:
            continue
        games[str(row.get("game_id", "?"))].append(Path(name).stem)
    for stems in games.values():
        stems.sort()
    return dict(games)


def sample_stratified(
    games: dict[str, list[str]], n: int, seed: int = config.SEED
) -> list[str]:
    """Take n stems spread as evenly as possible across games.

    Round-robins through the games so that small n still covers several
    distinct boards rather than several near-identical positions from one.
    """
    rng = random.Random(seed)
    pools = {g: rng.sample(s, len(s)) for g, s in sorted(games.items())}
    order = sorted(pools)

    picked: list[str] = []
    i = 0
    while len(picked) < n and any(pools[g] for g in order):
        g = order[i % len(order)]
        if pools[g]:
            picked.append(pools[g].pop())
        i += 1
    return picked


def build_subset(
    stems: Sequence[str],
    src_root: Path,
    split: str,
    dest: Path,
    class_names: Sequence[str],
) -> Path:
    """Materialise a symlinked training subset with its own data.yaml."""
    img_out = dest / "train" / "images"
    lbl_out = dest / "train" / "labels"
    for d in (img_out, lbl_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    src_img = src_root / split / "images"
    src_lbl = src_root / split / "labels"

    linked = 0
    for stem in stems:
        matches = list(src_img.glob(f"{stem}.*"))
        lbl = src_lbl / f"{stem}.txt"
        if not matches or not lbl.is_file():
            continue
        img = matches[0]
        (img_out / img.name).symlink_to(img.resolve())
        (lbl_out / lbl.name).symlink_to(lbl.resolve())
        linked += 1

    if linked == 0:
        sys.exit(f"No images linked into {dest}")

    # Validate on the VAL split, never test. Pointing this at test would let
    # early stopping and best-checkpoint selection see the test set, which
    # biases every number reported from it.
    yaml = dest / "data.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {dest.resolve()}",
                "train: train/images",
                f"val: {(src_root / 'val' / 'images').resolve()}",
                f"nc: {len(class_names)}",
                "names:",
                *(f"  - {n}" for n in class_names),
            ]
        )
        + "\n"
    )
    return yaml


# ---------------------------------------------------------------------------
# Run one point on the curve
# ---------------------------------------------------------------------------


def evaluate(model, data_yaml: Path, imgsz: int, device: str,
             split: str = "test") -> dict[str, float]:
    """Score a model on a held-out split.

    Defaults to "test" so results are comparable with the zero-shot baseline,
    which was measured on the 306-image test split. The val split is reserved
    for early stopping during fine-tuning and must not be reported.
    """
    res = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz,
        device=device, verbose=False, plots=False,
    )
    box = getattr(res, "box", None)

    def s(attr: str) -> float:
        try:
            return float(getattr(box, attr))
        except (TypeError, ValueError, AttributeError):
            return float("nan")

    return {"mAP50-95": s("map"), "mAP50": s("map50"),
            "precision": s("mp"), "recall": s("mr")}


def run_point(args, n: int, meta_rows, class_names) -> dict[str, Any]:
    from ultralytics import YOLO
    from train_yolo import resolve_device

    device = resolve_device(args.device)
    test_yaml = args.data / "data.yaml"

    if n == 0:
        print(f"\n{'=' * 62}\nN = 0  (zero-shot, no fine-tuning)\n{'=' * 62}")
        model = YOLO(str(args.weights))
        metrics = evaluate(model, test_yaml, args.imgsz, device)
        return {"n_images": 0, "epochs": 0, **metrics}

    games = stems_by_game(meta_rows, "train")
    total = sum(len(v) for v in games.values())
    take = total if n < 0 else min(n, total)

    print(f"\n{'=' * 62}\nN = {take}  ({len(games)} games available, "
          f"{total} train images)\n{'=' * 62}")

    stems = (sorted(s for v in games.values() for s in v)
             if n < 0 else sample_stratified(games, take, args.seed))

    work = args.data / "_finetune" / f"n{take}"
    work.mkdir(parents=True, exist_ok=True)
    subset_yaml = build_subset(stems, args.data, "train", work, class_names)

    covered = len({g for g, v in games.items() for s in v if s in set(stems)})
    print(f"  sampled {len(stems)} images spanning {covered} games")

    batch = max(1, min(args.batch, take))
    epochs = args.epochs
    name = f"ft-n{take}-{datetime.now():%Y%m%d-%H%M%S}"

    model = YOLO(str(args.weights))
    model.train(
        data=str(subset_yaml.resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=device,
        project=str(args.project),
        name=name,
        seed=args.seed,
        deterministic=True,
        # Adapting, not learning from scratch: a low LR avoids destroying the
        # representation the synthetic pre-training produced.
        lr0=args.lr0,
        lrf=0.05,
        warmup_epochs=1.0,
        freeze=args.freeze,
        patience=args.patience,
        val=True,
        plots=False,
        verbose=False,
        **config.YOLO_AUG,
    )

    best = args.project / name / "weights" / "best.pt"
    metrics = evaluate(YOLO(str(best)), test_yaml, args.imgsz, device)
    return {"n_images": take, "epochs": epochs, "games_covered": covered,
            "run": name, **metrics}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("runs/best.pt"),
                   help="Synthetic-trained starting checkpoint")
    p.add_argument("--data", type=Path,
                   default=config.PROCESSED_DIR / "chessred2k")
    p.add_argument("--n-images", type=int, default=None,
                   help="Number of real images to fine-tune on (-1 = all)")
    p.add_argument("--sweep", action="store_true",
                   help=f"Run the full curve: {DEFAULT_SWEEP}")
    p.add_argument("--points", type=str, default=None,
                   help="Comma-separated N values, e.g. 2,5,10,25,50")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds, e.g. 42,43,44. Each seed "
                        "changes which images are sampled AND the training "
                        "init, so the spread captures both sources of variance.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    p.add_argument("--lr0", type=float, default=1e-3,
                   help="Low by default: this is adaptation, not training")
    p.add_argument("--freeze", type=int, default=None,
                   help="Freeze the first N layers (e.g. 10 for the backbone)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--project", type=Path, default=config.RUNS_DIR / "finetune")
    p.add_argument("--out", type=Path,
                   default=config.REPORTS_DIR / "finetune_sweep.csv")
    args = p.parse_args()

    if not args.weights.is_file():
        sys.exit(f"No weights at {args.weights}")
    meta = args.data / "render_metadata.csv"
    if not meta.is_file():
        sys.exit(f"No metadata at {meta}")

    if args.points:
        points = [int(v) for v in args.points.split(",") if v.strip()]
    elif args.sweep:
        points = list(DEFAULT_SWEEP)
    elif args.n_images is not None:
        points = [args.n_images]
    else:
        sys.exit("Pass --n-images N, --points A,B,C, or --sweep")

    seeds = ([int(v) for v in args.seeds.split(",") if v.strip()]
             if args.seeds else [args.seed])

    with meta.open() as fh:
        meta_rows = list(csv.DictReader(fh))
    class_names = list(config.CANONICAL_CLASSES)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    total_runs = len(points) * len(seeds)
    done = 0

    for seed in seeds:
        args.seed = seed
        for n in points:
            done += 1
            if len(seeds) > 1:
                print(f"\n### run {done}/{total_runs}  (seed {seed}) ###")
            try:
                row = run_point(args, n, meta_rows, class_names)
            except Exception as exc:
                print(f"[error] N={n} seed={seed}: {exc}")
                continue
            row["seed"] = seed
            results.append(row)
            print(f"  -> mAP50-95 {row['mAP50-95']:.4f}   mAP50 {row['mAP50']:.4f}"
                  f"   P {row['precision']:.4f}   R {row['recall']:.4f}")

            # Merge with anything already on disk, keyed by (N, seed), so
            # separate invocations accumulate instead of overwriting and a
            # crash never loses completed work.
            fields = ["n_images", "seed", "epochs", "games_covered", "run",
                      "mAP50-95", "mAP50", "precision", "recall"]
            merged: dict[tuple[int, int], dict[str, Any]] = {}
            if args.out.is_file():
                with args.out.open() as fh:
                    for old in csv.DictReader(fh):
                        try:
                            key = (int(old["n_images"]),
                                   int(old.get("seed") or config.SEED))
                            merged[key] = old
                        except (KeyError, ValueError):
                            continue
            for r in results:
                merged[(int(r["n_images"]), int(r.get("seed", config.SEED)))] = r

            with args.out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for key in sorted(merged, key=lambda k: (k[0] < 0, k[0], k[1])):
                    w.writerow(merged[key])

    if not results:
        return 1

    # Summarise from the file, so points from earlier invocations are included.
    from statistics import mean, stdev

    by_n: dict[int, list[float]] = defaultdict(list)
    with args.out.open() as fh:
        for row in csv.DictReader(fh):
            try:
                by_n[int(row["n_images"])].append(float(row["mAP50-95"]))
            except (KeyError, ValueError):
                continue

    print(f"\n{'=' * 62}\nDATA EFFICIENCY CURVE\n{'=' * 62}")
    multi = any(len(v) > 1 for v in by_n.values())
    if multi:
        print(f"{'N':>8} {'seeds':>6} {'mean':>9} {'std':>8} "
              f"{'min':>8} {'max':>8} {'spread':>8}")
        print("-" * 56)
    else:
        print(f"{'N':>8} {'mAP50-95':>10}")
        print("-" * 20)

    for n in sorted(by_n, key=lambda k: (k < 0, k)):
        vals = by_n[n]
        if multi:
            sd = stdev(vals) if len(vals) > 1 else 0.0
            print(f"{n:>8} {len(vals):>6} {mean(vals):>9.4f} {sd:>8.4f} "
                  f"{min(vals):>8.4f} {max(vals):>8.4f} "
                  f"{max(vals)-min(vals):>8.4f}")
        else:
            print(f"{n:>8} {vals[0]:>10.4f}")

    if 0 in by_n:
        base = mean(by_n[0])
        rest = {n: mean(v) for n, v in by_n.items() if n != 0}
        if rest:
            best_n = max(rest, key=rest.get)
            print(f"\n  zero-shot: {base:.4f}")
            print(f"  best mean: {rest[best_n]:.4f} at N={best_n}")
            if base > 0:
                print(f"  improvement: {rest[best_n] / base:.1f}x")

    if multi:
        spreads = [max(v) - min(v) for v in by_n.values() if len(v) > 1]
        if spreads:
            print(f"\n  largest spread across seeds: {max(spreads):.4f}")
            print("  differences smaller than this should not be treated")
            print("  as meaningful.")

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
