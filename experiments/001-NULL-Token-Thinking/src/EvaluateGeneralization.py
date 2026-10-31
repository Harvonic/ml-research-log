"""Batched greedy evaluation and input-only NULL ablations; no weight changes."""
import argparse
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from ExperimentUtils import (SRC, ROOT, DEVICE, read_lines, load_model, encode,
                             carry_count, forward_flops, write_csv)


@torch.inference_mode()
def evaluate(model, vocab, lines, count, token, batch_size):
    loss_sum, target_count = 0.0, 0
    for start in range(0, len(lines), batch_size):
        X, Y, _ = encode(lines[start:start + batch_size], vocab, count, token)
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        logits = model(X)
        loss_sum += F.cross_entropy(logits.flatten(0, 1), Y.flatten(),
                                   ignore_index=-100, reduction="sum").item()
        target_count += (Y != -100).sum().item()

    # Equal-length prompt buckets avoid left padding and preserve original positions.
    groups = defaultdict(list)
    for index, line in enumerate(lines):
        problem, answer = line.split(" = ")
        ids = [vocab[c] for c in problem + " = "]
        if count:
            ids += [vocab[token]] * count
        groups[len(ids)].append((index, ids, answer))
    reverse = {i: t for t, i in vocab.items()}
    forbidden = [vocab[t] for t in ("<PAD>", "<UNK>", "<NULL>") if t in vocab]
    hits, by_carry, inference, executed = 0, defaultdict(lambda: [0, 0]), 0, 0
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            ids = torch.tensor([row[1] for row in batch], device=DEVICE)
            done = torch.zeros(len(batch), dtype=torch.bool, device=DEVICE)
            outputs = [[] for _ in batch]
            for _ in range(24):
                inference += forward_flops(int((~done).sum()), ids.shape[1], len(vocab))
                executed += forward_flops(len(batch), ids.shape[1], len(vocab))
                logits = model(ids)[:, -1].clone()
                logits[:, forbidden] = -float("inf")
                nxt = logits.argmax(-1)
                for i, (value, finished) in enumerate(zip(nxt.tolist(), done.tolist())):
                    if not finished and value != vocab["<EOS>"]:
                        outputs[i].append(reverse[value])
                done |= nxt == vocab["<EOS>"]
                if done.all():
                    break
                ids = torch.cat((ids, nxt[:, None]), dim=1)
            for i, (index, _, answer) in enumerate(batch):
                prediction = "".join(outputs[i]) + ("..." if not bool(done[i]) else "")
                hit = int(prediction == answer)
                hits += hit
                carry = carry_count(lines[index])
                by_carry[carry][0] += hit
                by_carry[carry][1] += 1
    return dict(examples=len(lines), correct=hits, accuracy=hits / len(lines),
                answer_loss=loss_sum / target_count, answer_tokens=target_count,
                inference_flops_per_problem=inference / len(lines),
                batched_inference_flops_per_problem=executed / len(lines)), by_carry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / "results" / ("flop_matched" if args.matched else "generalization")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    test = read_lines(SRC / "data" / "test.txt")
    datasets = {5: random.Random(42).sample(test, min(10000, len(test)))}
    datasets.update({d: read_lines(SRC / "data" / "generalization" / f"addition_{d}_digit.txt") for d in (6, 7)})
    if args.matched:
        configs = [("baseline_matched", ROOT / "results/flop_matched/baseline_matched.pt", 0, "<NULL>")]
    else:
        configs = [("baseline", ROOT / "results/baseline.pt", 0, "<NULL>")]
        configs += [(f"null_{n}", ROOT / "results/null.pt", n, "<NULL>") for n in (0, 1, 2, 3, 4, 8)]
        configs += [("replacement_4", ROOT / "results/null.pt", 4, "<UNK>")]
    rows, carries, errors = [], [], []
    for name, path, count, token in configs:
        model, vocab = load_model(path)
        for digits, lines in datasets.items():
            try:
                metrics, by_carry = evaluate(model, vocab, lines, count, token, args.batch_size)
            except Exception as exc:
                error = f"{name}, {digits} digits: {type(exc).__name__}: {exc}"
                errors.append(error)
                (out / "logs" / "errors.txt").write_text("\n".join(errors), encoding="utf-8")
                raise
            row = dict(configuration=name, digits=digits, null_count=count, extra_token=token,
                       parameters=sum(p.numel() for p in model.parameters()), **metrics)
            rows.append(row)
            carries.extend(dict(configuration=name, digits=digits, carries=c, correct=h,
                                examples=n, accuracy=h / n) for c, (h, n) in sorted(by_carry.items()))
            write_csv(out / "metrics" / "metrics.csv", rows)
            write_csv(out / "metrics" / "carry_metrics.csv", carries)
            print(f"{name:18s} {digits} digits: accuracy={metrics['accuracy']:.4%}, loss={metrics['answer_loss']:.6f}", flush=True)
        del model
    print(f"Metrics saved to {out}", flush=True)


if __name__ == "__main__":
    main()
