"""Render extension plots exclusively into experiment-specific plots/ folders."""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ExperimentUtils import ROOT

RESULTS = ROOT / "results"
COLORS = ["#4477AA", "#EE6677", "#228833"]


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def grouped(rows, configs, title, directory, filename):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs, width = np.arange(3), 0.8 / len(configs)
    for i, (config, label) in enumerate(configs):
        vals = [100 * float(next(r for r in rows if r["configuration"] == config and int(r["digits"]) == d)["accuracy"])
                for d in (5, 6, 7)]
        bars = ax.bar(xs + (i - (len(configs) - 1) / 2) * width, vals, width,
                      label=label, color=COLORS[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set(xticks=xs, xticklabels=["Original (up to 5)", "6 digits", "7 digits"],
           ylabel="Exact-answer accuracy (%)", xlabel="Operand length", ylim=(0, 114), title=title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    save(fig, directory, filename)


def main():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    gen = RESULTS / "generalization"
    rows = read_csv(gen / "metrics" / "metrics.csv")
    directory = gen / "plots"
    grouped(rows, [("baseline", "Baseline"), ("null_4", "NULL × 4")],
            "Arithmetic length generalization", directory, "generalization_accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, color, marker in zip((5, 6, 7), COLORS, ("o", "s", "^")):
        counts = (0, 1, 2, 3, 4, 8)
        vals = [100 * float(next(r for r in rows if r["configuration"] == f"null_{n}" and int(r["digits"]) == d)["accuracy"])
                for n in counts]
        ax.plot(counts, vals, marker=marker, color=color, markerfacecolor="none",
                markersize=6 + 2 * (d - 5), linestyle={5: "-", 6: "--", 7: ":"}[d],
                label="Original (up to 5 digits)" if d == 5 else f"{d} digits")
    ax.text(.03, .28, "6- and 7-digit accuracy:\n0% at every NULL count", transform=ax.transAxes,
            ha="left", fontsize=9, color="#555555")
    ax.axvline(4, color="gray", linestyle="--", alpha=.6, label="Training count")
    ax.set(xlabel="Inserted NULL tokens", ylabel="Exact-answer accuracy (%)", ylim=(-3, 108),
           xticks=counts, title="Inference-time NULL-count ablation")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.2)
    save(fig, directory, "null_count_ablation")
    grouped(rows, [("null_4", "NULL × 4"), ("replacement_4", "UNK × 4"), ("null_0", "No extra tokens")],
            "NULL embedding / position interventions", directory, "embedding_ablation")

    # Read old logs without modifying them; PowerShell may have written UTF-16.
    matched = RESULTS / "flop_matched"
    budgets = {r["model"]: r for r in read_csv(matched / "metrics" / "compute_budget.csv")}
    raw = (RESULTS / "training_log.txt").read_bytes()
    text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    current, last_epoch, last_step, last_avg = None, 0, 0, 0.
    traces = {name: [] for name in ("baseline", "null")}
    for line in text.splitlines():
        line = line.strip()
        if match := re.match(r"^(baseline|null):", line):
            current, last_epoch, last_step, last_avg = match[1], 0, 0, 0.
        elif current and (match := re.match(r"epoch (\d+) \| step (\d+) \| avg loss ([\d.eE+-]+)", line)):
            epoch, step, avg = int(match[1]), int(match[2]), float(match[3])
            if epoch != last_epoch:
                last_step, last_avg = 0, 0.
            window = (avg * step - last_avg * last_step) / (step - last_step)
            budget = budgets[current]
            global_step = (epoch - 1) * (int(budget["steps"]) // 5) + step
            traces[current].append((global_step * float(budget["mean_training_flops_per_step"]) / 1e12, window))
            last_epoch, last_step, last_avg = epoch, step, avg
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, (name, trace) in enumerate(traces.items()):
        if trace:
            x, y = zip(*[(x, y) for x, y in trace if y > 0])
            ax.plot(x, y, color=COLORS[i], label="Original " + ("NULL" if name == "null" else name),
                    alpha=.85, linewidth=2.5 if name == "baseline" else 1.5)
    if (matched / "metrics" / "training_metrics.csv").exists():
        training = read_csv(matched / "metrics" / "training_metrics.csv")
        ax.plot([float(r["cumulative_flops"]) / 1e12 for r in training],
                [float(r["window_mean_loss"]) for r in training], color=COLORS[2],
                linewidth=1.5, linestyle="--", label="FLOP-matched baseline")
    ax.set(xlabel="Cumulative estimated training FLOPs (trillions)",
           ylabel="Mean answer-token training loss per logged interval", yscale="log",
           title="Training loss versus estimated compute")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, matched / "plots", "training_efficiency")
    (matched / "reports").mkdir(parents=True, exist_ok=True)
    (matched / "reports" / "plot_notes.txt").write_text(
        "Original efficiency curves recover interval means from rounded epoch-cumulative console logs.\n"
        "Their FLOP axis uses reconstructed mean padded step cost; all original shuffled batches have the same maximum length.\n"
        "Matched curve uses directly logged interval means and cumulative FLOPs.\n"
        "Five-digit label refers to the original up-to-five-digit distribution; each evaluation has 10,000 cases.\n", encoding="utf-8")
    if (matched / "metrics" / "metrics.csv").exists():
        matched_rows = read_csv(matched / "metrics" / "metrics.csv")
        combined = [r for r in rows if r["configuration"] in ("baseline", "null_4")] + matched_rows
        for row in combined:
            name = {"null_4": "null"}.get(row["configuration"], row["configuration"])
            budget = budgets[name]
            row.update(training_steps=budget["steps"], training_flops=budget["estimated_training_flops"],
                       accuracy_per_training_tflop=float(row["accuracy"]) / (float(budget["estimated_training_flops"]) / 1e12))
        from ExperimentUtils import write_csv
        write_csv(matched / "metrics" / "comparison.csv", combined)
        grouped(combined, [("baseline", "Original baseline"), ("null_4", "Original NULL × 4"),
                          ("baseline_matched", "FLOP-matched baseline")],
                "Arithmetic accuracy at matched training compute", matched / "plots", "flop_matched_comparison")
    summary(rows, matched)
    print("Saved experiment plots in results/generalization/plots and results/flop_matched/plots.")


def summary(rows, matched):
    lines = ["NULL-token generalization and ablation results", "",
             "Exact-answer accuracy (%); each cell uses 10,000 examples.",
             "Configuration             Original <=5      6 digits      7 digits"]
    for config in dict.fromkeys(r["configuration"] for r in rows):
        values = [next(r for r in rows if r["configuration"] == config and int(r["digits"]) == d) for d in (5, 6, 7)]
        lines.append(f"{config:24s}" + "".join(f"{float(r['accuracy']):14.2%}" for r in values))
    lines += ["", "Original evaluation sample is seed-42, matching Train.py; not all million test rows.",
              "Answer loss is token-weighted teacher-forced cross-entropy, including EOS and excluding prompt/extra tokens.",
              "Primary accuracy uses greedy generation, requires correct EOS termination, and uses each checkpoint's own vocabulary.",
              "NULL/PAD/UNK answer logits are forbidden exactly as in the original predictor.",
              "Ablations modify only prompt inputs. NULL checkpoint always retains its original vocabulary and weights.",
              "Counts other than four, and UNK replacement, are inference-time distribution shifts.",
              "Removal also shifts answer positions; replacement holds length fixed but is OOD.",
              "A drop under either intervention cannot establish latent reasoning or isolate the effect of sequence positions.",
              "No position-only control was trained, so extra computation and extra positions remain confounded.",
              "Longer inputs also reach learned absolute positions beyond the maximum used in original training; this possible source of failure is not isolated.",
              "See reports/data_audit.txt for leakage; metrics/carry_metrics.csv for carry-stratified counts/accuracies."]
    original = [r for r in rows if r["configuration"] in ("baseline", "null_4") and int(r["digits"]) > 5]
    if original and all(float(r["accuracy"]) == 0 for r in original):
        lines.append("Both original checkpoints score zero exact matches on the new lengths: no evidence of arithmetic length generalization.")
    (RESULTS / "generalization/summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not (matched / "metrics" / "comparison.csv").exists():
        return
    compare = read_csv(matched / "metrics" / "comparison.csv")
    lines = ["FLOP-matched evaluation", "", "configuration,digits,accuracy,answer_loss,parameters,training_steps,estimated_training_flops"]
    for r in compare:
        lines.append(",".join(r[k] for k in ("configuration", "digits", "accuracy", "answer_loss", "parameters", "training_steps", "training_flops")))
    lines += ["", "Training from scratch used the original seed, data order, initialization, optimizer, learning rate and architecture.",
              "See metrics/actual_training.csv for actual steps, examples, compute and mismatch; reports/flop_method.txt for estimator limitations.",
              "Accuracy per training TFLOP is descriptive, not a universal efficiency measure: accuracy is bounded and nonlinear in compute.",
              "Inference FLOPs are reported in metrics/comparison.csv: full-prefix greedy forwards without a KV cache, through EOS.",
              "inference_flops_per_problem counts only active problems; batched_inference_flops_per_problem also includes finished rows.",
              "Failed OOD answers can terminate early and appear cheaper; inference cost must be read alongside accuracy.",
              "Training matching leaves additional NULL inference cost and vocabulary/position differences.",
              "Only one initialization seed was evaluated; no across-seed significance claim is supported.",
              "These controls do not establish learned reasoning or isolate positional effects."]
    actual = read_csv(matched / "metrics" / "actual_training.csv")[0]
    lines += ["", f"Actual matched steps: {int(actual['steps']):,}; examples: {int(actual['examples']):,}",
              f"Estimated matched training FLOPs: {float(actual['estimated_training_flops']):.8e}",
              f"NULL training target FLOPs: {float(actual['target_flops']):.8e}",
              f"Relative budget mismatch: {float(actual['relative_mismatch']):.9%}",
              "", "Original-distribution inference (mean estimated dense FLOPs/problem):"]
    for r in compare:
        if r["digits"] == "5":
            lines.append(f"{r['configuration']}: {float(r['inference_flops_per_problem']):,.0f}")
    five = {r["configuration"]: float(r["accuracy"]) for r in compare if r["digits"] == "5"}
    if five["null_4"] > five["baseline_matched"]:
        lines.append("In this single run the NULL model retains a same-distribution accuracy advantage over the compute-matched baseline.")
    else:
        lines.append("In this single run the NULL model does not outperform the compute-matched baseline on same-distribution exact-answer accuracy.")
    (matched / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
