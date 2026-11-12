"""Small JSON and plotting helpers for the training sweep."""

import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def plot_run(metrics, path):
    history = metrics["history"]
    steps = [row["step"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, [row["train_loss"] for row in history], label="Train (interval mean)")
    axes[0].plot(steps, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(steps, [100 * row["val_accuracy"] for row in history])
    axes[1].set_ylabel("Validation next-token accuracy (%)")
    axes[2].plot(steps, [row["training_tokens_per_second"] for row in history])
    axes[2].set_ylabel("Training tokens/second")
    for ax in axes:
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.25)
    config = metrics["config"]
    fig.suptitle(f"XSA | lambda={config['lam']:g} | seed={config['seed']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_comparison(runs, directory):
    # Aggregate only completed runs from the current sweep.
    fields = ("final_val_loss", "final_val_accuracy", "training_tokens_per_second", "elapsed_seconds")
    summary = []
    for lam in dict.fromkeys(run["config"]["lam"] for run in runs):
        selected = [run for run in runs if run["config"]["lam"] == lam]
        row = {"lam": lam, "seeds": [run["config"]["seed"] for run in selected]}
        for field in fields:
            values = [run["summary"][field] for run in selected]
            row[field] = {
                "mean": round(statistics.mean(values), 6),
                "std": round(statistics.stdev(values), 6) if len(values) > 1 else None,
            }
        summary.append(row)
    write_json(directory / "summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = [f"{row['lam']:g}\n(n={len(row['seeds'])})" for row in summary]
    for ax, field, title, scale in zip(
        axes, fields[:3],
        ("Final validation loss", "Final next-token accuracy (%)", "Training tokens/second"),
        (1, 100, 1),
    ):
        means = [row[field]["mean"] * scale for row in summary]
        errors = [(row[field]["std"] or 0) * scale for row in summary]
        ax.bar(range(len(summary)), means, yerr=errors, capsize=4)
        ax.set_xticks(range(len(summary)), labels)
        ax.set_xlabel("Lambda (completed seeds)")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("XSA comparison | mean and sample standard deviation across seeds")
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)
