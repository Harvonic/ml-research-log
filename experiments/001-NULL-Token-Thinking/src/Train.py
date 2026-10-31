import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from Model import Transformer

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

TRAIN_FILE = SRC / "data" / "train.txt"
TEST_FILE = SRC / "data" / "test.txt"
OUT_DIR = ROOT / "results"

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NULL_COUNT = 4
TEST_SAMPLES = 10000
MAX_ANSWER_TOKENS = 24
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SRC / "vocab.json", encoding="utf-8") as f:
    saved_vocab = json.load(f)

tokens = sorted(saved_vocab, key=saved_vocab.get)

with open(TRAIN_FILE, encoding="utf-8") as f:
    train_lines = [line.strip() for line in f if line.strip()]
with open(TEST_FILE, encoding="utf-8") as f:
    test_lines = [line.strip() for line in f if line.strip()]
test_cases = random.Random(SEED).sample(test_lines, min(TEST_SAMPLES, len(test_lines)))


def make_batch(lines, vocab, null_count):
    inputs, targets = [], []
    for line in lines:
        problem, answer = line.split(" = ", 1)
        prompt = [vocab[c] for c in problem + " = "]
        answer_ids = [vocab[c] for c in answer] + [vocab["<EOS>"]]
        nulls = [vocab["<NULL>"]] * null_count if null_count else []

        # Train only on answer tokens, NOT on the question or NULL tokens.
        x = (prompt + nulls + answer_ids)[:-1]
        y = [-100] * (len(prompt) + len(nulls) - 1) + answer_ids
        inputs.append(x)
        targets.append(y)

    length = max(map(len, inputs))
    X = torch.full((len(lines), length), vocab["<PAD>"], dtype=torch.long)
    Y = torch.full((len(lines), length), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(inputs, targets)):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X.to(DEVICE), Y.to(DEVICE)


@torch.inference_mode()
def predict(model, problem, vocab, null_count):
    ids = [vocab[c] for c in problem + " = "]
    if null_count:
        ids += [vocab["<NULL>"]] * null_count
    reverse = {i: token for token, i in vocab.items()}
    forbidden = [vocab[t] for t in ("<PAD>", "<UNK>", "<NULL>") if t in vocab]
    output = []

    for _ in range(MAX_ANSWER_TOKENS):
        X = torch.tensor([ids], device=DEVICE)
        logits = model(X)[0, -1].clone()
        logits[forbidden] = -float("inf")
        next_id = int(logits.argmax())
        if next_id == vocab["<EOS>"]:
            return "".join(output)
        output.append(reverse[next_id])
        ids.append(next_id)

    return "".join(output) + "..."  # Failed to stop with EOS


def run(name, null_count):
    vocab_tokens = tokens if null_count else [t for t in tokens if t != "<NULL>"]
    vocab = {token: i for i, token in enumerate(vocab_tokens)}
    torch.manual_seed(SEED)
    model = Transformer(len(vocab)).to(DEVICE)
    # Weight tying with default nn.Embedding initialization gives huge initial logits.
    torch.nn.init.normal_(model.token_emb.weight, std=0.02)
    torch.nn.init.normal_(model.pos_emb.weight, std=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\n{name}: {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        indices = torch.randperm(len(train_lines), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        total_loss, steps = 0.0, 0
        for start in range(0, len(indices), BATCH_SIZE):
            batch = [train_lines[i] for i in indices[start:start + BATCH_SIZE]]
            X, Y = make_batch(batch, vocab, null_count)
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), Y.reshape(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            if steps % 500 == 0:
                print(f"  epoch {epoch + 1} | step {steps} | avg loss {total_loss / steps:.4f}")
        print(f"  epoch {epoch + 1} finished | avg loss {total_loss / steps:.4f}")

    model.eval()
    test_loss, token_count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(test_cases), BATCH_SIZE):
            X, Y = make_batch(test_cases[start:start + BATCH_SIZE], vocab, null_count)
            logits = model(X)
            test_loss += F.cross_entropy(
                logits.reshape(-1, len(vocab)), Y.reshape(-1),
                ignore_index=-100, reduction="sum"
            ).item()
            token_count += (Y != -100).sum().item()
    print(f"  Held-out answer loss: {test_loss / token_count:.4f}")

    correct = 0
    per_op = {op: [0, 0] for op in ["+", "-", "*", "/"]}
    for i, line in enumerate(test_cases):
        problem, answer = line.split(" = ", 1)
        result = predict(model, problem, vocab, null_count)
        hit = int(result == answer)
        correct += hit
        op = problem.split()[1]
        per_op[op][0] += hit
        per_op[op][1] += 1
        if i < 5:
            print(f"  {problem} = {result}  (expected {answer})")

    print(f"  Exact-answer accuracy: {correct}/{len(test_cases)} = {correct / len(test_cases):.2%}")
    print("  By operation:", ", ".join(
        f"{op}: {hits / n:.1%}" for op, (hits, n) in per_op.items() if n
    ))
    torch.save({"weights": model.state_dict(), "vocab": vocab, "null_count": null_count},
               OUT_DIR / f"{name}.pt")


if __name__ == "__main__":
    run("baseline", 0)
    run("null", NULL_COUNT)
