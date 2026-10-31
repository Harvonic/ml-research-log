"""Fresh baseline run matching analytical padded training FLOPs of original NULL."""
import argparse
import csv
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from ExperimentUtils import (SRC, ROOT, DEVICE, Transformer, read_lines,
                             encode, forward_flops, write_csv)

OUT = ROOT / "results" / "flop_matched"


def order(epoch, size):
    return torch.randperm(size, generator=torch.Generator().manual_seed(42 + epoch))


def step_lengths(lengths, indices):
    return [int(lengths[indices[start:start + 64]].max())
            for start in range(0, len(indices), 64)]


def estimate(lines, baseline_vocab, null_vocab):
    lengths = np.array([len(line) for line in lines], dtype=np.int16)
    n, per_epoch = len(lines), math.ceil(len(lines) / 64)
    records, null_target = [], 0
    for name, extra, vocab_size in (("baseline", 0, baseline_vocab), ("null", 4, null_vocab)):
        cumulative = 0
        for epoch in range(5):
            lens = step_lengths(lengths + extra, order(epoch, n).numpy())
            for step, length in enumerate(lens):
                batch = min(64, n - step * 64)
                cumulative += 3 * forward_flops(batch, length, vocab_size)
        records.append(dict(model=name, steps=5 * per_epoch, examples=5 * n,
                            estimated_training_flops=cumulative,
                            mean_training_flops_per_step=cumulative / (5 * per_epoch)))
        if name == "null":
            null_target = cumulative
    cumulative, steps, examples, epoch = 0, 0, 0, 0
    while True:
        lens = step_lengths(lengths, order(epoch, n).numpy())
        for step, length in enumerate(lens):
            batch = min(64, n - step * 64)
            flops = 3 * forward_flops(batch, length, baseline_vocab)
            if cumulative + flops >= null_target:
                if abs(cumulative + flops - null_target) < abs(cumulative - null_target):
                    cumulative, steps, examples = cumulative + flops, steps + 1, examples + batch
                records.append(dict(model="baseline_matched", steps=steps, examples=examples,
                                    estimated_training_flops=cumulative,
                                    mean_training_flops_per_step=cumulative / steps))
                return records, null_target
            cumulative, steps, examples = cumulative + flops, steps + 1, examples + batch
        epoch += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    for directory in ("metrics", "logs", "reports"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    lines = read_lines(SRC / "data/train.txt")
    original = torch.load(ROOT / "results/baseline.pt", map_location="cpu", weights_only=True)
    null = torch.load(ROOT / "results/null.pt", map_location="cpu", weights_only=True)
    vocab = original["vocab"]
    records, target = estimate(lines, len(vocab), len(null["vocab"]))
    write_csv(OUT / "metrics" / "compute_budget.csv", records)
    total_steps = records[-1]["steps"]
    assumptions = [
        "Analytical dense-matmul estimate; one multiply-add = two FLOPs.",
        "Per layer forward: 8BLd^2 (QKV/output) + 4BL^2d (QK/AV) + 8BLd^2 (FFN width 2d).",
        "Head: 2BLdV. d=72, layers=2. Training FLOPs=3*forward (approximate backward).",
        "Full square attention is counted despite causal mask. No fused-kernel profiling used.",
        "Excludes biases, normalization, activations, softmax/CE, optimizer, clipping, embedding lookup and memory traffic.",
        "Uses actual maximum sequence length in each shuffled batch, including final partial batches.",
        "Sequences match Train.py: answer plus EOS shifted once; padded to batch maximum.",
        "Replays torch.randperm with separate CPU Generator(seed=42+epoch), batch size 64.",
        "Original run FLOPs are reconstructed from current code/data, not measured historical hardware counters.",
        "Closest whole optimizer step to the five-epoch NULL budget is used.",
        "Matching this estimator does not exactly match wall time, energy, or all GPU operations.",
        "New run uses seed 42, float32, default AdamW(lr=3e-4), clip_grad_norm=1, no LR schedule.",
        "Pre-encoding and device-resident tensors change data transfer only, preserving batch membership and targets.",
        f"Device: {DEVICE}; PyTorch: {torch.__version__}",
        f"Target NULL training FLOPs: {target}; matched baseline steps: {total_steps}",
        f"Planned relative mismatch: {(records[-1]['estimated_training_flops'] / target - 1):.9%}",
    ]
    (OUT / "reports" / "flop_method.txt").write_text("\n".join(assumptions) + "\n", encoding="utf-8")
    print("\n".join(assumptions[-3:]), flush=True)
    if args.estimate_only:
        return
    checkpoint_path = OUT / "baseline_matched.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite {checkpoint_path}")
    random.seed(42)
    torch.manual_seed(42)
    model = Transformer(len(vocab)).to(DEVICE)
    torch.nn.init.normal_(model.token_emb.weight, std=0.02)
    torch.nn.init.normal_(model.pos_emb.weight, std=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print("Pre-encoding original training data...", flush=True)
    X, Y, lengths = encode(lines, vocab)
    del lines
    lengths = np.array(lengths)
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    steps, examples, cumulative, epoch = 0, 0, 0, 0
    started = time.monotonic()
    with (OUT / "logs" / "training_log.txt").open("w", encoding="utf-8", buffering=1) as log, \
         (OUT / "metrics" / "training_metrics.csv").open("w", newline="", encoding="utf-8", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "step", "epoch_step", "examples",
            "cumulative_flops", "epoch_mean_loss", "window_mean_loss", "elapsed_seconds"])
        writer.writeheader()
        def report(message):
            print(message, flush=True)
            log.write(message + "\n")
        report(f"Fresh seed-42 baseline on {DEVICE}; target={target}; steps={total_steps}")
        while steps < total_steps:
            indices = order(epoch, len(X))
            lens = step_lengths(lengths, indices.numpy())
            indices = indices.to(DEVICE)
            epoch_loss, window_loss, window_steps = 0.0, 0.0, 0
            model.train()
            for j, start in enumerate(range(0, len(X), 64)):
                batch_indices = indices[start:start + 64]
                length = lens[j]
                xb = X.index_select(0, batch_indices)[:, :length]
                yb = Y.index_select(0, batch_indices)[:, :length]
                logits = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, len(vocab)), yb.reshape(-1), ignore_index=-100)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                value = loss.item()
                epoch_loss += value
                window_loss += value
                window_steps += 1
                steps += 1
                examples += len(batch_indices)
                cumulative += 3 * forward_flops(len(batch_indices), length, len(vocab))
                if (j + 1) % 500 == 0 or j + 1 == len(lens) or steps == total_steps:
                    elapsed = time.monotonic() - started
                    writer.writerow(dict(epoch=epoch + 1, step=steps, epoch_step=j + 1, examples=examples,
                        cumulative_flops=cumulative, epoch_mean_loss=epoch_loss / (j + 1),
                        window_mean_loss=window_loss / window_steps, elapsed_seconds=elapsed))
                    report(f"epoch {epoch + 1} | total step {steps}/{total_steps} | avg loss {epoch_loss/(j+1):.6f} | {elapsed:.1f}s")
                    window_loss, window_steps = 0.0, 0
                if steps == total_steps:
                    break
            epoch += 1
        torch.save(dict(weights=model.state_dict(), vocab=vocab, null_count=0,
                        training_steps=steps, examples_processed=examples,
                        estimated_training_flops=cumulative, target_flops=target, seed=42), checkpoint_path)
        report(f"Saved {checkpoint_path}; mismatch={cumulative / target - 1:.9%}")
    write_csv(OUT / "metrics" / "actual_training.csv", [dict(steps=steps, examples=examples,
        estimated_training_flops=cumulative, target_flops=target,
        relative_mismatch=cumulative / target - 1, elapsed_seconds=time.monotonic() - started)])


if __name__ == "__main__":
    main()
