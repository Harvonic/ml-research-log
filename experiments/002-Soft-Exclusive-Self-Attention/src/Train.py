import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

from Model import Transformer
from Results import plot_run, save_comparison, write_json


# Every seed runs for one lambda before the next lambda starts.
# LAM_VALUES = [0, 0.1, 0.25, 0.5, 1, 1.5, 2]

# Second run using more extreme values, just for fun
LAM_VALUES = [-2, -1, 1.25, 2.5, 3, 5]
SEEDS = [5, 12, 23, 42, 71]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = 192
LAYERS = 4
HEADS = 4
CONTEXT = 256
BATCH_SIZE = 32
STEPS = 5000
LR = 3e-4
EVAL_INTERVAL = 250
EVAL_BATCHES = 20

DATASET = "roneneldan/TinyStories"
TRAIN_SPLIT = "train[:100000]"
VAL_SPLIT = "validation[:2000]"
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
TOKENIZER_PATH = ROOT / "src" / "vocab.json"


def load_tokens():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("The tokenizer must contain [EOS].")

    def tokenize(split):
        tokens = []
        for example in load_dataset(DATASET, split=split):
            tokens.extend(tokenizer.encode(example["text"]).ids)
            tokens.append(eos_id)
        return torch.tensor(tokens, dtype=torch.long)

    print("Loading and tokenizing TinyStories once for the entire sweep...")
    train_tokens = tokenize(TRAIN_SPLIT)
    val_tokens = tokenize(VAL_SPLIT)
    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")
    return train_tokens, val_tokens, tokenizer.get_vocab_size()


def get_batch(tokens, generator):
    ix = torch.randint(0, len(tokens) - CONTEXT, (BATCH_SIZE,), generator=generator)
    positions = ix[:, None] + torch.arange(CONTEXT)
    return tokens[positions].to(DEVICE), tokens[positions + 1].to(DEVICE)


def forward_loss(model, x, y):
    with torch.autocast(
        device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"
    ):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    return logits, loss


@torch.no_grad()
def evaluate(model, tokens, seed):
    # Fixed validation windows; evaluation never advances the training RNG.
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    total_loss = 0.0
    correct = 0
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(tokens, generator)
        logits, loss = forward_loss(model, x, y)
        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
    model.train()
    return total_loss / EVAL_BATCHES, correct / (EVAL_BATCHES * BATCH_SIZE * CONTEXT)


def synchronize():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def run_directory(lam):
    return RESULTS_DIR / f"xsa-lam={lam:g}"


def train_run(lam, seed, train_tokens, val_tokens, vocab_size):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_generator = torch.Generator().manual_seed(seed)

    run_dir = run_directory(lam)
    json_path = run_dir / "resultsJson" / f"seed_{seed}" / "metrics.json"
    visual_path = run_dir / "visuals" / f"seed_{seed}" / "metrics.png"
    checkpoint_path = run_dir / "checkpoints" / f"seed_{seed}" / "model.pt"
    model = Transformer(
        vocab_size=vocab_size, d=D, layers=LAYERS, heads=HEADS,
        context=CONTEXT, lam=lam
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    metrics = {
        "status": "running",
        "config": {
            "lam": lam, "seed": seed, "device": DEVICE,
            "device_name": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
            "torch_version": str(torch.__version__),
            "d": D, "layers": LAYERS, "heads": HEADS, "context": CONTEXT,
            "vocab_size": vocab_size, "parameters": sum(p.numel() for p in model.parameters()),
            "batch_size": BATCH_SIZE, "steps": STEPS, "learning_rate": LR,
            "weight_decay": 0.01, "gradient_clip": 1.0, "amp": DEVICE == "cuda",
            "eval_interval": EVAL_INTERVAL, "eval_batches": EVAL_BATCHES,
            "dataset": DATASET, "train_split": TRAIN_SPLIT, "val_split": VAL_SPLIT,
            "train_tokens": len(train_tokens), "val_tokens": len(val_tokens),
        },
        "summary": {},
        "history": [],
    }
    write_json(json_path, metrics)
    print(f"Lambda {lam:g} | Seed {seed} | Parameters: {metrics['config']['parameters']:,}")
    model.train()
    synchronize()
    started = time.perf_counter()
    interval_started = started
    training_seconds = 0.0
    interval_loss = 0.0
    interval_steps = 0
    best_loss = float("inf")
    best_step = 0

    for step in range(1, STEPS + 1):
        x, y = get_batch(train_tokens, train_generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = forward_loss(model, x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        interval_loss += loss.item()
        interval_steps += 1
        if step % EVAL_INTERVAL != 0 and step != STEPS:
            continue

        synchronize()
        interval_seconds = time.perf_counter() - interval_started
        training_seconds += interval_seconds
        val_loss, val_accuracy = evaluate(model, val_tokens, seed)
        if not math.isfinite(val_loss) or not math.isfinite(interval_loss):
            raise RuntimeError(f"Non-finite loss at lambda={lam}, seed={seed}, step={step}.")
        if val_loss < best_loss:
            best_loss, best_step = val_loss, step
        record = {
            "step": step,
            "train_loss": round(interval_loss / interval_steps, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "training_tokens_per_second": round(
                interval_steps * BATCH_SIZE * CONTEXT / interval_seconds, 2
            ),
        }
        metrics["history"].append(record)
        metrics["summary"] = {
            "completed_steps": step,
            "final_train_loss": record["train_loss"],
            "final_val_loss": record["val_loss"],
            "best_val_loss": round(best_loss, 6), "best_step": best_step,
            "final_val_accuracy": record["val_accuracy"],
            "final_val_perplexity": round(math.exp(val_loss), 4) if val_loss < 700 else None,
            "elapsed_seconds": record["elapsed_seconds"],
            "training_seconds": round(training_seconds, 3),
            "training_tokens_per_second": round(step * BATCH_SIZE * CONTEXT / training_seconds, 2),
        }
        write_json(json_path, metrics)
        print(
            f"  Step {step:5d}/{STEPS} | Train: {record['train_loss']:.4f} | "
            f"Val: {val_loss:.4f} | Accuracy: {val_accuracy:.2%} | "
            f"TPS: {record['training_tokens_per_second']:,.0f}"
        )
        interval_loss, interval_steps = 0.0, 0
        synchronize()
        interval_started = time.perf_counter()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    plot_run(metrics, visual_path)
    metrics["summary"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    metrics["status"] = "completed"
    write_json(json_path, metrics)
    return metrics


def main():
    if not LAM_VALUES or not SEEDS:
        raise ValueError("LAM_VALUES and SEEDS must both contain at least one value.")
    if any(not isinstance(lam, (int, float)) or not math.isfinite(lam) for lam in LAM_VALUES):
        raise ValueError("Every lambda must be a finite number.")
    if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in SEEDS):
        raise ValueError("Seeds must be integers between 0 and 2**32 - 1.")
    if len({f"{lam:g}" for lam in LAM_VALUES}) != len(LAM_VALUES) or len(set(SEEDS)) != len(SEEDS):
        raise ValueError("Use unique lambda values and seeds (including their folder names).")
    if min(STEPS, BATCH_SIZE, CONTEXT, EVAL_INTERVAL, EVAL_BATCHES) < 1:
        raise ValueError("Steps, batch size, context and evaluation settings must be positive.")

    # Check before loading data so existing runs are never silently overwritten.
    for lam in LAM_VALUES:
        for seed in SEEDS:
            for folder in ("resultsJson", "visuals", "checkpoints"):
                path = run_directory(lam) / folder / f"seed_{seed}"
                if path.exists():
                    raise FileExistsError(f"Existing run: {path}. Move it or change RESULTS_DIR before rerunning.")

    train_tokens, val_tokens, vocab_size = load_tokens()
    if min(len(train_tokens), len(val_tokens)) <= CONTEXT:
        raise ValueError("Both token streams must be longer than CONTEXT.")
    runs = []
    total = len(LAM_VALUES) * len(SEEDS)
    for lam in LAM_VALUES:
        for seed in SEEDS:
            print(f"\nRun {len(runs) + 1}/{total}")
            runs.append(train_run(lam, seed, train_tokens, val_tokens, vocab_size))
            save_comparison(runs, RESULTS_DIR / "comparison")
    print(f"\nCompleted {total} runs. Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
