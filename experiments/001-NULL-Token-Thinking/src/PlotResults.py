"""Plot baseline vs NULL training from Train.py console logs.

Usage:
    python src/PlotResults.py results/training_2026-10-31_15-30-00.txt

Without an argument, reads the newest training_*.txt in ../results/.
Requires: pip install matplotlib
"""

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

if len(sys.argv) > 1:
    log_file = Path(sys.argv[1])
else:
    candidates = list((ROOT / "results").glob("training_*.txt"))
    if not candidates:
        raise SystemExit("No training log found. Pass its path: python PlotResults.py path/to/log.txt")
    log_file = max(candidates, key=lambda p: p.stat().st_mtime)

# Match the exact print format in Train.py.
model_re = re.compile(r"^(baseline|null):\s+[\d,]+\s+params\b", re.I)
step_re = re.compile(r"^epoch (\d+) \| step (\d+) \| avg loss ([\d.eE+-]+)$")
epoch_re = re.compile(r"^epoch (\d+) finished \| avg loss ([\d.eE+-]+)$")
accuracy_re = re.compile(r"^Exact-answer accuracy:.*?= ([\d.]+)%$")
heldout_re = re.compile(r"^Held-out answer loss: ([\d.eE+-]+)$")

results = {name: {"epochs": [], "windows": [], "accuracy": None, "test_loss": None}
           for name in ("baseline", "null")}
active = None
last_epoch, last_step, last_avg = None, 0, 0.0

# PowerShell 5.1 Tee-Object commonly saves UTF-16 LE logs.
# Decode UTF-8 and UTF-16 logs automatically.
raw_bytes = log_file.read_bytes()
if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
    log_text = raw_bytes.decode("utf-16")
elif raw_bytes[:200].count(b"\x00") >= 5:
    log_text = raw_bytes.decode("utf-16-le")
else:
    log_text = raw_bytes.decode("utf-8-sig", errors="replace")

for raw in log_text.splitlines():
    line = raw.strip()
    match = model_re.match(line)
    if match:
        active = match.group(1).lower()
        last_epoch, last_step, last_avg = None, 0, 0.0
        continue
    if active is None:
        continue

    if match := step_re.match(line):
        epoch, step, avg = int(match[1]), int(match[2]), float(match[3])
        if epoch != last_epoch:
            last_epoch, last_step, last_avg = epoch, 0, 0.0
        # Logs show a cumulative mean within the epoch. Recover the mean loss
        # over the newest 500-step interval (approximately, due to rounding).
        window_mean = (avg * step - last_avg * last_step) / (step - last_step)
        results[active]["windows"].append((epoch, step, window_mean))
        last_step, last_avg = step, avg
    elif match := epoch_re.match(line):
        results[active]["epochs"].append((int(match[1]), float(match[2])))
    elif match := accuracy_re.match(line):
        results[active]["accuracy"] = float(match[1])
    elif match := heldout_re.match(line):
        results[active]["test_loss"] = float(match[1])

if not any(r["epochs"] for r in results.values()):
    print(f"Read {len(log_text.splitlines())} lines from {log_file}, but no matching epoch losses.")
    print("First few nonempty lines:")
    for line in [ln.strip() for ln in log_text.splitlines() if ln.strip()][:8]:
        print(repr(line[:180]))
    raise SystemExit("Expected 'baseline: ... params' then 'epoch 1 finished | avg loss 0.1234'.")

# 1. Training loss by epoch.
plt.figure(figsize=(8, 5))
for name, r in results.items():
    if r["epochs"]:
        x, y = zip(*r["epochs"])
        plt.plot(x, y, marker="o", linewidth=2, label=name.upper())
plt.xlabel("Epoch")
plt.ylabel("Average answer-token cross-entropy")
plt.title("Arithmetic training loss — Baseline vs NULL")
plt.xticks(sorted({x for r in results.values() for x, _ in r["epochs"]}))
plt.yscale("log")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "loss_by_epoch.png", dpi=180)
plt.close()

# 2. Approximate mean loss for each 500-step interval.
# The x-axis is expressed in epochs to avoid assuming a particular batch size.
plt.figure(figsize=(9, 5))
for name, r in results.items():
    if not r["windows"]:
        continue
    max_step = max(step for _, step, _ in r["windows"])
    points = [(epoch - 1 + step / max_step, loss)
              for epoch, step, loss in r["windows"] if loss > 0]
    if points:
        x, y = zip(*points)
        plt.plot(x, y, linewidth=1.5, label=name.upper())
plt.xlabel("Training progress (approximate epochs)")
plt.ylabel("Mean answer-token loss per logged step interval")
plt.title("Training loss progression — 500-step intervals")
plt.yscale("log")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "loss_by_step.png", dpi=180)
plt.close()

# 3. Held-out exact-answer accuracy, if present in the log.
acc = [(name.upper(), r["accuracy"]) for name, r in results.items()
       if r["accuracy"] is not None]
if acc:
    plt.figure(figsize=(6, 5))
    bars = plt.bar([name for name, _ in acc], [value for _, value in acc])
    for bar, (_, value) in zip(bars, acc):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1,
                 f"{value:.2f}%", ha="center")
    plt.ylim(0, 105)
    plt.ylabel("Exact-answer accuracy (%)")
    plt.title("Held-out arithmetic accuracy")
    plt.tight_layout()
    plt.savefig(OUT / "test_accuracy.png", dpi=180)
    plt.close()

print(f"Read: {log_file}")
for name, r in results.items():
    if r["epochs"]:
        print(f"{name}: final training loss={r['epochs'][-1][1]:.4f}, "
              f"test loss={r['test_loss']}, accuracy={r['accuracy']}%")
print(f"Plots saved to: {OUT}")
