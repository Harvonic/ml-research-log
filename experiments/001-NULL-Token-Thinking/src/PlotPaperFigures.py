"""Render a column-width paper figure from the saved comparison metrics."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main():
    with (ROOT / "results/flop_matched/metrics/comparison.csv").open(newline="") as f:
        rows = {r["configuration"]: r for r in csv.DictReader(f) if r["digits"] == "5"}
    names = ("baseline", "baseline_matched", "null_4")
    errors = [int(rows[n]["examples"]) - int(rows[n]["correct"]) for n in names]
    out = ROOT / "results/flop_matched/plots/paper"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.05), layout="constrained")
    bars = ax.bar(["Baseline", "Matched\nbaseline", "NULL × 4"], errors,
                  color=["#4477AA", "#228833", "#CC6677"], width=.62, zorder=3)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set(ylabel="Exact-answer errors / 10,000", ylim=(0, 520), yticks=(0, 100, 200, 300, 400, 500))
    ax.grid(axis="y", alpha=.2, zorder=0)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"exact_answer_errors.{extension}", dpi=300)
    plt.close(fig)
    print(f"Paper figure saved to {out}")


if __name__ == "__main__":
    main()
