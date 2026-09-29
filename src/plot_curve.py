"""Plot the data-efficiency curve from reports/finetune_sweep.csv.

Points measured with more than one seed are drawn with error bars spanning
min to max across seeds, so the reader can see directly where the curve is
trustworthy and where it is not. Single-seed points are drawn hollow, since
a lone measurement carries no information about its own spread.

Usage
-----
    python src/plot_curve.py --out ../domain_gap_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("reports/finetune_sweep.csv"))
    p.add_argument("--out", type=Path, default=Path("../domain_gap_curve.png"))
    p.add_argument("--synthetic-ceiling", type=float, default=0.9366,
                   help="mAP50-95 of the synthetic-trained model on renders")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(args.csv)
    g = (df.groupby("n_images")["mAP50-95"]
           .agg(["mean", "min", "max", "count"])
           .reset_index()
           .sort_values("n_images"))

    zero = g[g.n_images == 0]["mean"].iloc[0]
    curve = g[g.n_images > 0]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    multi = curve[curve["count"] > 1]
    single = curve[curve["count"] == 1]

    ax.plot(curve.n_images, curve["mean"], "-", color="#1f4e79", lw=2, zorder=2)
    ax.errorbar(multi.n_images, multi["mean"],
                yerr=[multi["mean"] - multi["min"], multi["max"] - multi["mean"]],
                fmt="o", color="#1f4e79", ms=7, capsize=5, lw=1.5, zorder=3,
                label="3 seeds (bars = min to max)")
    ax.plot(single.n_images, single["mean"], "o", mfc="white",
            mec="#1f4e79", mew=1.8, ms=7, zorder=3, label="1 seed only")

    ax.axhline(zero, color="#c0392b", ls="--", lw=1.4,
               label=f"zero-shot on real photos ({zero:.3f})")
    ax.axhline(args.synthetic_ceiling, color="#27ae60", ls=":", lw=1.4,
               label=f"synthetic-domain ceiling ({args.synthetic_ceiling:.3f})")

    ax.set_xscale("log")
    ax.set_xticks(curve.n_images)
    ax.set_xticklabels([str(int(n)) for n in curve.n_images])
    ax.minorticks_off()
    ax.set_xlabel("Real photographs used for fine-tuning (log scale)")
    ax.set_ylabel("mAP50-95 on the 306-image real test set")
    ax.set_title("Recovering the synthetic-to-real domain gap")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    print("\n     N  seeds     mean     spread")
    for _, r in g.iterrows():
        print(f"{int(r.n_images):6d} {int(r['count']):6d}   "
              f"{r['mean']:.4f}   {r['max'] - r['min']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
