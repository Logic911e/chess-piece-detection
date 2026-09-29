"""ResNet-50 classification baseline on piece crops.

The design report proposed a ResNet-50 baseline. The project became a detection
study, so a like-for-like comparison is not possible: a classifier cannot find
pieces and a detector's mAP is not an accuracy. What *is* comparable, and more
informative, is the domain gap each one suffers.

The classifier is handed ground-truth boxes, so localisation is free and only
naming the piece is left. Measuring its synthetic-to-real drop therefore
isolates the classification half of the problem, which is exactly what finding
F7 claimed the domain shift damages. If F7 is right, this baseline should lose
a large fraction of its accuracy across the domain shift even though it never
had to localise anything.

Three decompositions are reported because the 12-way accuracy alone hides the
interesting structure:

  * type accuracy (6-way: which piece) versus colour accuracy (2-way), since
    F3 and F6 found colour behaves very differently from shape;
  * per-class recall, since the classes are severely imbalanced (pawns are
    over half of all instances and queens about 2.5%);
  * balanced accuracy and macro F1, which do not let the pawn class carry the
    score.

Counting convention. `--limit-images N` samples N *source photographs*, not N
crops, stratified across games. This is deliberate: the detector's curve is
indexed by photographs, and a comparison indexed by crops would silently give
the classifier thirty times the data at every point.

Usage
-----
    # train on synthetic crops, test on both domains
    python src/resnet_baseline.py \\
        --train data/crops/synth/train --val data/crops/synth/val \\
        --eval synthetic=data/crops/synth/test \\
        --eval real=data/crops/chessred2k/test

    # fine-tune that model on N real photographs
    python src/resnet_baseline.py --init runs/resnet/synth/best.pt \\
        --train data/crops/chessred2k/train --limit-images 25 \\
        --val data/crops/chessred2k/val \\
        --eval real=data/crops/chessred2k/test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import config


def pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def source_key(filename: str) -> str:
    """Recover the source photograph stem from a crop filename.

    Crops are named `<source-stem>_<index>.jpg`, so dropping the final numeric
    suffix recovers the photograph a crop came from.
    """
    return re.sub(r"_\d+$", "", Path(filename).stem)


def game_key(stem: str) -> str:
    """Group photographs by game, so a small sample spans distinct boards.

    ChessReD names images `G000_IMG014`; anything without that structure falls
    back to being its own group, which degrades gracefully to plain sampling.
    """
    return stem.split("_")[0] if "_" in stem else stem


def subset_by_images(dataset, n_images: int, seed: int):
    """Restrict an ImageFolder to crops from n source photographs.

    Photographs are drawn round-robin across games, so N=2 takes two different
    boards rather than two consecutive moves of the same position. This mirrors
    the sampling in finetune.py so the two curves are comparable.
    """
    from torch.utils.data import Subset

    by_game: dict[str, set[str]] = defaultdict(set)
    for path, _ in dataset.samples:
        stem = source_key(Path(path).name)
        by_game[game_key(stem)].add(stem)

    rng = random.Random(seed)
    pools = []
    for game in sorted(by_game):
        stems = sorted(by_game[game])
        rng.shuffle(stems)
        pools.append(stems)
    rng.shuffle(pools)

    chosen: list[str] = []
    i = 0
    while len(chosen) < n_images and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            chosen.append(pool.pop())
        else:
            pools = [p for p in pools if p]
            if not pools:
                break
            continue
        i += 1

    keep = set(chosen)
    idx = [k for k, (path, _) in enumerate(dataset.samples)
           if source_key(Path(path).name) in keep]
    games = {game_key(s) for s in keep}
    print(f"  sampled {len(keep)} photographs spanning {len(games)} games "
          f"-> {len(idx)} crops")
    return Subset(dataset, idx), len(keep)


def build_loaders(args, device: str):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        norm,
    ])
    # No horizontal flip. A chess piece is near-symmetric so a flip is
    # harmless for most classes, but knights face a direction and flipping
    # them is the one augmentation that could plausibly hurt.

    check_class_dirs(args.train)
    train_ds = datasets.ImageFolder(args.train, transform=train_tf)
    check_classes(train_ds.classes, args.train)

    n_images = None
    if args.limit_images:
        train_ds, n_images = subset_by_images(train_ds, args.limit_images, args.seed)

    kw = dict(num_workers=args.workers, pin_memory=(device == "cuda"))
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=len(train_ds) > args.batch, **kw)

    val_ld = None
    if args.val:
        check_class_dirs(args.val)
        val_ds = datasets.ImageFolder(args.val, transform=eval_tf)
        check_classes(val_ds.classes, args.val)
        val_ld = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, **kw)

    evals = {}
    for spec in args.eval or []:
        if "=" not in spec:
            sys.exit(f"--eval expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        check_class_dirs(Path(path))
        ds = datasets.ImageFolder(Path(path), transform=eval_tf)
        check_classes(ds.classes, path)
        evals[name] = DataLoader(ds, batch_size=args.batch * 2, shuffle=False, **kw)

    return train_ld, val_ld, evals, n_images


def check_class_dirs(root: Path) -> None:
    """Fail clearly on missing or empty class directories.

    ImageFolder derives its label indices from the directory listing, so a
    class folder that is absent shifts every later label by one and a class
    folder that is empty raises a FileNotFoundError whose message does not say
    which class caused it. Both are worth catching before a long run starts
    rather than an hour into one.
    """
    if not root.is_dir():
        sys.exit(f"Dataset not found: {root}")
    missing, empty = [], []
    for name in config.CANONICAL_CLASSES:
        d = root / name
        if not d.is_dir():
            missing.append(name)
        elif not any(d.iterdir()):
            empty.append(name)
    if missing or empty:
        sys.exit(f"Unusable crop dataset at {root}\n"
                 + (f"  missing class directories: {missing}\n" if missing else "")
                 + (f"  empty class directories: {empty}\n" if empty else "")
                 + "  Re-run make_crops.py, or check --min-side is not "
                   "discarding every instance of a class.")


def check_classes(found: list[str], where) -> None:
    """Abort on any class-ordering mismatch.

    ImageFolder sorts directory names alphabetically, which is *not* the order
    of config.CANONICAL_CLASSES. Every dataset is therefore checked to contain
    exactly the expected names; the label index used throughout is ImageFolder's
    own, consistently, so nothing depends on the two orders agreeing. A missing
    or extra class would silently shift every label, so it is fatal.
    """
    expected = sorted(config.CANONICAL_CLASSES)
    if list(found) != expected:
        missing = set(expected) - set(found)
        extra = set(found) - set(expected)
        sys.exit(f"Class mismatch in {where}\n"
                 f"  missing: {sorted(missing)}\n  unexpected: {sorted(extra)}")


def build_model(classes: list[str], init: Path | None, device: str, args):
    import torch
    from torchvision import models

    weights = None if init else models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    if init:
        state = torch.load(init, map_location="cpu")
        saved = state.get("classes")
        if saved and list(saved) != list(classes):
            sys.exit(f"Checkpoint classes differ from dataset classes:\n"
                     f"  checkpoint: {saved}\n  dataset:    {classes}")
        model.load_state_dict(state["model"])
        print(f"initialised from {init}")

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.") or name.startswith("layer4")
        print("froze all layers except layer4 and fc")

    return model.to(device)


def evaluate(model, loader, classes: list[str], device: str) -> dict:
    """Run the model over a loader and summarise with metrics_from_confusion."""
    import torch

    model.eval()
    n = len(classes)
    confusion = [[0] * n for _ in range(n)]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t][p] += 1

    return metrics_from_confusion(confusion, classes)


def metrics_from_confusion(confusion: list[list[int]], classes: list[str]) -> dict:
    """Accuracy, the type/colour decomposition, and per-class figures.

    Kept free of torch so it can be tested directly. `confusion[t][p]` is the
    number of instances of true class t predicted as class p.
    """
    n = len(classes)
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(n))
    if total == 0:
        return {}

    # Per-class recall and precision from the confusion matrix.
    recalls, f1s = [], []
    per_class = {}
    for i, name in enumerate(classes):
        support = sum(confusion[i])
        tp = confusion[i][i]
        predicted = sum(confusion[r][i] for r in range(n))
        rec = tp / support if support else 0.0
        prec = tp / predicted if predicted else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {"support": support, "recall": rec,
                           "precision": prec, "f1": f1}
        if support:
            recalls.append(rec)
            f1s.append(f1)

    # Decompose the 12-way prediction. "white-bishop" -> colour white, type
    # bishop, so a prediction can be right about the shape and wrong about the
    # colour, or the reverse. Those are different failures and F3/F6 suggest
    # they behave differently under domain shift.
    type_ok = colour_ok = 0
    for i, tname in enumerate(classes):
        t_colour, t_type = tname.split("-", 1)
        for j, pname in enumerate(classes):
            p_colour, p_type = pname.split("-", 1)
            c = confusion[i][j]
            if t_type == p_type:
                type_ok += c
            if t_colour == p_colour:
                colour_ok += c

    return {
        "n": total,
        "top1": correct / total,
        "balanced_acc": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "type_acc": type_ok / total,
        "colour_acc": colour_ok / total,
        "per_class": per_class,
        "confusion": confusion,
    }


def train(model, train_ld, val_ld, classes, device, args, out_dir: Path):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    weight = None
    if args.class_weights:
        counts = Counter(y for _, y in _iter_targets(train_ld.dataset))
        total = sum(counts.values())
        w = [total / (len(classes) * max(1, counts.get(i, 0)))
             for i in range(len(classes))]
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("using inverse-frequency class weights")
    crit = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = seen = 0
        for i, (x, y) in enumerate(train_ld, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            seen += y.numel()
            if i % 50 == 0:
                print(f"  epoch {epoch} [{i}/{len(train_ld)}] "
                      f"loss {running / seen:.4f}", flush=True)
        sched.step()

        line = {"epoch": epoch, "loss": running / max(1, seen),
                "secs": round(time.time() - t0, 1)}
        if val_ld is not None:
            m = evaluate(model, val_ld, classes, device)
            line |= {"val_top1": m["top1"], "val_balanced": m["balanced_acc"]}
            score = m["balanced_acc"]
        else:
            score = -line["loss"]

        history.append(line)
        print(f"epoch {epoch}/{args.epochs}  " +
              "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                        for k, v in line.items() if k != "epoch"), flush=True)

        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "classes": classes,
                        "epoch": epoch, "score": score}, out_dir / "best.pt")

    # Selection uses the validation split only. The test sets passed with
    # --eval are never consulted during training, for the same reason the
    # detector's fine-tuning validates on val: choosing a checkpoint on test
    # would leak it into every number reported from it.
    if (out_dir / "best.pt").is_file():
        state = torch.load(out_dir / "best.pt", map_location="cpu")
        model.load_state_dict(state["model"])
        print(f"\nrestored best checkpoint from epoch {state['epoch']}")
    return history


def _iter_targets(ds):
    from torch.utils.data import Subset

    if isinstance(ds, Subset):
        for i in ds.indices:
            yield None, ds.dataset.samples[i][1]
    else:
        for _, t in ds.samples:
            yield None, t


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, default=None)
    p.add_argument("--eval", action="append", metavar="NAME=PATH",
                   help="Test set to report on; repeatable")
    p.add_argument("--init", type=Path, default=None,
                   help="Start from this checkpoint instead of ImageNet")
    p.add_argument("--out", type=Path, default=Path("runs/resnet"))
    p.add_argument("--tag", default=None, help="Subdirectory name for this run")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit-images", type=int, default=None,
                   help="Train on crops from only N source photographs")
    p.add_argument("--class-weights", action="store_true",
                   help="Weight the loss by inverse class frequency")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Train only layer4 and fc")
    args = p.parse_args()

    try:
        import torch
        import torchvision  # noqa: F401
    except ImportError:
        sys.exit("torch and torchvision required: pip install torch torchvision")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"device: {device}")
    if device == "mps":
        # Same caveat as the detector: Apple's backend has no deterministic
        # implementation for several reductions, so runs are not bit-identical
        # even with a fixed seed. Reported spreads include that.
        print("note: MPS is not bit-reproducible; repeat with several seeds "
              "before trusting small differences")

    train_ld, val_ld, evals, n_images = build_loaders(args, device)
    base = train_ld.dataset
    classes = (base.dataset.classes if hasattr(base, "dataset") else base.classes)

    tag = args.tag or (f"n{args.limit_images}" if args.limit_images else "full")
    out_dir = args.out / f"{tag}-seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(classes, args.init, device, args)
    n_train = len(train_ld.dataset)
    print(f"\ntrain crops: {n_train}"
          + (f" from {n_images} photographs" if n_images else "")
          + f"\nclasses: {len(classes)}\nepochs: {args.epochs}\n")

    history = train(model, train_ld, val_ld, classes, device, args, out_dir)

    report = {"tag": tag, "seed": args.seed, "device": device,
              "train_crops": n_train, "train_images": n_images,
              "epochs": args.epochs, "init": str(args.init) if args.init else None,
              "history": history, "results": {}}

    print("\n" + "=" * 62)
    print(f"{'set':<12}{'n':>7}{'top1':>9}{'balanced':>10}"
          f"{'macroF1':>9}{'type':>8}{'colour':>8}")
    print("-" * 62)
    for name, loader in evals.items():
        m = evaluate(model, loader, classes, device)
        report["results"][name] = m
        print(f"{name:<12}{m['n']:>7}{m['top1']:>9.4f}{m['balanced_acc']:>10.4f}"
              f"{m['macro_f1']:>9.4f}{m['type_acc']:>8.4f}{m['colour_acc']:>8.4f}")
    print("=" * 62)

    if len(report["results"]) >= 2 and {"synthetic", "real"} <= report["results"].keys():
        s = report["results"]["synthetic"]["balanced_acc"]
        r = report["results"]["real"]["balanced_acc"]
        print(f"\nclassification domain gap: {s:.4f} -> {r:.4f}  "
              f"({(s - r) / s * 100:.1f}% of balanced accuracy lost)")
        report["domain_gap"] = {"synthetic": s, "real": r,
                                "relative_loss": (s - r) / s if s else None}

    for name, m in report["results"].items():
        rows = sorted(m["per_class"].items(),
                      key=lambda kv: kv[1]["recall"], reverse=True)
        print(f"\nper-class recall on {name}:")
        for cls, v in rows:
            print(f"  {cls:<14}{v['recall']:>7.3f}   (n={v['support']})")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    for name, m in report["results"].items():
        with (out_dir / f"confusion_{name}.csv").open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["true\\pred", *classes])
            for cls, row in zip(classes, m["confusion"]):
                wr.writerow([cls, *row])

    print(f"\nwrote {out_dir}/report.json and confusion matrices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
