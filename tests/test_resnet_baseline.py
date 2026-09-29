"""Tests for the torch-free parts of the ResNet-50 baseline.

The metric code is the part most likely to be wrong in a way that still looks
plausible: a transposed confusion matrix, or a type/colour decomposition that
quietly counts the diagonal twice, produces numbers that are internally
consistent and completely false. Each is checked against a case whose answer
can be worked out by hand.

Run with:  python tests/test_resnet_baseline.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

# The module imports torch lazily inside functions, but `subset_by_images`
# imports Subset at call time, so a minimal stub stands in for it.
if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")

    class Subset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

    data.Subset = Subset
    utils.data = data
    torch.utils = utils
    sys.modules["torch"] = torch
    sys.modules["torch.utils"] = utils
    sys.modules["torch.utils.data"] = data

import resnet_baseline as rb  # noqa: E402
import config  # noqa: E402

FAILURES: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-9
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(label)


class FakeFolder:
    """Stands in for torchvision ImageFolder: only `.samples` is used."""

    def __init__(self, samples):
        self.samples = samples
        self.classes = sorted(config.CANONICAL_CLASSES)


def test_source_and_game_keys() -> None:
    print("\nfilename parsing")
    check("crop -> source photograph",
          rb.source_key("G003_IMG017_042.jpg"), "G003_IMG017")
    check("index stripped only once",
          rb.source_key("G003_IMG017_007.jpg"), "G003_IMG017")
    check("game grouping", rb.game_key("G003_IMG017"), "G003")
    # A name with no underscore must not crash or collapse everything into
    # one group; it becomes its own group, degrading to plain sampling.
    check("ungrouped name is its own game", rb.game_key("render0421"), "render0421")


def test_subset_spans_games() -> None:
    """N photographs must come from N different games where possible.

    This is the property that makes the classifier curve comparable to the
    detector's: if N=2 drew two consecutive moves of one game, the two boards
    would be nearly identical and the result would flatter the model.
    """
    print("\nsampling N photographs across games")
    samples = []
    for g in range(4):
        for img in range(10):
            for crop in range(3):
                samples.append(
                    (f"/x/G{g:03d}_IMG{img:03d}_{crop:03d}.jpg", crop % 12))
    ds = FakeFolder(samples)

    subset, n_images = rb.subset_by_images(ds, 4, seed=1)
    check("photograph count", n_images, 4)
    check("crop count is 3 per photograph", len(subset.indices), 12)

    stems = {rb.source_key(Path(ds.samples[i][0]).name) for i in subset.indices}
    games = {rb.game_key(s) for s in stems}
    check("four photographs span four games", len(games), 4)

    # Asking for more photographs than exist must return everything rather
    # than loop forever or raise.
    subset_all, n_all = rb.subset_by_images(ds, 999, seed=1)
    check("over-request is clamped to the pool", n_all, 40)
    check("over-request keeps every crop", len(subset_all.indices), 120)


def test_subset_is_deterministic() -> None:
    print("\nsampling determinism")
    samples = [(f"/x/G{g:03d}_IMG{i:03d}_000.jpg", 0)
               for g in range(3) for i in range(8)]
    ds = FakeFolder(samples)
    a, _ = rb.subset_by_images(ds, 5, seed=7)
    b, _ = rb.subset_by_images(ds, 5, seed=7)
    c, _ = rb.subset_by_images(ds, 5, seed=8)
    check("same seed gives same sample", a.indices, b.indices)
    check("different seed gives different sample", a.indices != c.indices, True)


def test_metrics_perfect() -> None:
    print("\nmetrics: perfect classifier")
    classes = sorted(config.CANONICAL_CLASSES)
    n = len(classes)
    conf = [[5 if i == j else 0 for j in range(n)] for i in range(n)]
    m = rb.metrics_from_confusion(conf, classes)
    check("top1", m["top1"], 1.0)
    check("balanced accuracy", m["balanced_acc"], 1.0)
    check("macro F1", m["macro_f1"], 1.0)
    check("type accuracy", m["type_acc"], 1.0)
    check("colour accuracy", m["colour_acc"], 1.0)
    check("total counted", m["n"], 5 * n)


def test_metrics_colour_blind() -> None:
    """A classifier that always gets the shape right and the colour wrong.

    This is the case that catches a bad decomposition. Top-1 must be 0,
    type accuracy must be 1, colour accuracy must be 0. Any implementation
    that conflates the two will score these the same.
    """
    print("\nmetrics: right shape, wrong colour")
    classes = sorted(config.CANONICAL_CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    conf = [[0] * len(classes) for _ in classes]
    for c in classes:
        colour, ptype = c.split("-", 1)
        other = ("black" if colour == "white" else "white") + "-" + ptype
        conf[idx[c]][idx[other]] = 4
    m = rb.metrics_from_confusion(conf, classes)
    check("top1 is zero", m["top1"], 0.0)
    check("type accuracy is one", m["type_acc"], 1.0)
    check("colour accuracy is zero", m["colour_acc"], 0.0)


def test_metrics_imbalance() -> None:
    """Balanced accuracy must not be rescued by the majority class.

    With pawns dominating the real dataset, a model that predicts nothing but
    white-pawn scores respectably on top-1 and must score near zero on
    balanced accuracy. If it does not, the headline number is meaningless.
    """
    print("\nmetrics: majority-class-only classifier")
    classes = sorted(config.CANONICAL_CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    conf = [[0] * len(classes) for _ in classes]
    conf[idx["white-pawn"]][idx["white-pawn"]] = 90
    for c in classes:
        if c != "white-pawn":
            conf[idx[c]][idx["white-pawn"]] = 10

    m = rb.metrics_from_confusion(conf, classes)
    total = 90 + 10 * 11
    check("top1 reflects the majority", m["top1"], 90 / total)
    check("balanced accuracy is 1/12", m["balanced_acc"], 1 / 12)
    check("per-class recall on an unpredicted class",
          m["per_class"]["black-queen"]["recall"], 0.0)
    check("support is recorded per class",
          m["per_class"]["black-queen"]["support"], 10)


def test_metrics_orientation() -> None:
    """confusion[true][pred], not the transpose.

    A transposed matrix swaps precision and recall everywhere, which is
    invisible on symmetric test cases. This case is deliberately asymmetric.
    """
    print("\nmetrics: matrix orientation")
    classes = sorted(config.CANONICAL_CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    conf = [[0] * len(classes) for _ in classes]
    # Every black-queen is called a black-bishop; black-bishops are correct.
    conf[idx["black-queen"]][idx["black-bishop"]] = 20
    conf[idx["black-bishop"]][idx["black-bishop"]] = 60

    m = rb.metrics_from_confusion(conf, classes)
    check("queen recall is zero", m["per_class"]["black-queen"]["recall"], 0.0)
    check("bishop recall is one", m["per_class"]["black-bishop"]["recall"], 1.0)
    check("bishop precision is 60/80",
          m["per_class"]["black-bishop"]["precision"], 0.75)
    # The shape is wrong but the colour is right for all 20 errors.
    check("colour survives the confusion", m["colour_acc"], 1.0)
    check("type accuracy is 60/80", m["type_acc"], 0.75)


def test_metrics_empty() -> None:
    print("\nmetrics: empty input")
    classes = sorted(config.CANONICAL_CLASSES)
    conf = [[0] * len(classes) for _ in classes]
    check("empty confusion returns empty dict",
          rb.metrics_from_confusion(conf, classes), {})


def main() -> int:
    test_source_and_game_keys()
    test_subset_spans_games()
    test_subset_is_deterministic()
    test_metrics_perfect()
    test_metrics_colour_blind()
    test_metrics_imbalance()
    test_metrics_orientation()
    test_metrics_empty()

    print("\n" + "=" * 50)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
