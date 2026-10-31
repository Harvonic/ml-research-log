"""Shared extension helpers; the original model and training files stay untouched."""
import csv
import hashlib
import os
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
# Model.py reads vocab.json relative to cwd and creates two demonstration models.
_cwd = Path.cwd()
try:
    os.chdir(SRC)
    from Model import Transformer
finally:
    os.chdir(_cwd)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_lines(path):
    return [s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    weights, vocab = checkpoint["weights"], checkpoint["vocab"]
    # Infer saved dimensions; these checkpoints do not save head count (known: 4).
    d = weights["token_emb.weight"].shape[1]
    layers = len({key.split(".")[2] for key in weights if key.startswith("blocks.layers.")})
    model = Transformer(len(vocab), d=d, layers=layers, heads=4,
                        context=weights["pos_emb.weight"].shape[0])
    model.load_state_dict(weights, strict=True)
    return model.to(DEVICE).eval(), vocab


def encode(lines, vocab, count=0, extra_token="<NULL>"):
    inputs, targets, lengths = [], [], []
    for line in lines:
        problem, answer = line.split(" = ")
        prompt = [vocab[c] for c in problem + " = "]
        if count:
            prompt += [vocab[extra_token]] * count
        answer_ids = [vocab[c] for c in answer] + [vocab["<EOS>"]]
        inputs.append((prompt + answer_ids)[:-1])
        targets.append([-100] * (len(prompt) - 1) + answer_ids)
        lengths.append(len(inputs[-1]))
    X = torch.full((len(lines), max(lengths)), vocab["<PAD>"], dtype=torch.long)
    Y = torch.full_like(X, -100)
    # One allocation per row matches the original token/target construction.
    for i, (x, y) in enumerate(zip(inputs, targets)):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X, Y, lengths


def forward_flops(batch, length, vocab_size, d=72, layers=2):
    """Dense matmul FLOPs, multiply-add=2; full LxL attention, no causal saving.

    QKV+attention output: 8BLd²; QK+AV: 4BL²d;
    FFN (width 2d): 8BLd²; vocabulary projection: 2BLdV.
    Excludes normalization, activations, bias, softmax, embedding lookup,
    optimizer and memory traffic. Training uses 3x forward (forward+backward).
    """
    return batch * (layers * (16 * length * d * d + 4 * length * length * d)
                    + 2 * length * d * vocab_size)


def carry_count(line):
    problem = line.split(" = ")[0]
    a, b = map(int, problem.split(" + "))
    carry, total = 0, 0
    while a or b:
        carry = int(a % 10 + b % 10 + carry >= 10)
        total += carry
        a, b = a // 10, b // 10
    return total
