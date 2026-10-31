"""Generate separate, reproducible length-OOD data and audit existing datasets."""
import argparse
import random
from collections import Counter

from ExperimentUtils import SRC, ROOT, read_lines, sha256, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    out = ROOT / "results" / "generalization"
    (out / "reports").mkdir(parents=True, exist_ok=True)
    generated = []
    for digits in (6, 7):
        seed = 4200 + digits
        rng, seen, lines = random.Random(seed), set(), []
        low, high = 10 ** (digits - 1), 10 ** digits - 1
        while len(lines) < args.samples:
            a, b = rng.randint(low, high), rng.randint(low, high)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            lines.append(f"{a} + {b} = {a + b}")
        path = SRC / "data" / "generalization" / f"addition_{digits}_digit.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace differing dataset: {path}")
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        generated.append(dict(digits=digits, examples=len(lines), seed=seed, sha256=sha256(path)))
    write_csv(out / "metrics" / "datasets.csv", generated)

    train, test = (read_lines(SRC / "data" / f"{name}.txt") for name in ("train", "test"))
    train_set, test_set = set(train), set(test)
    overlap = train_set & test_set
    def unordered(line):
        a, b = map(int, line.split(" = ")[0].split(" + "))
        return min(a, b), max(a, b)
    unordered_train = {unordered(line) for line in train}
    counts = Counter()
    for name, lines in (("train", train), ("test", test)):
        for line in lines:
            problem, answer = line.split(" = ")
            a, b = map(int, problem.split(" + "))
            if a + b != int(answer):
                raise ValueError(f"Invalid {name} arithmetic: {line}")
            counts[(name, len(str(a)), len(str(b)))] += 1
    write_csv(out / "metrics" / "operand_lengths.csv", [dict(dataset=n, a_digits=a, b_digits=b, examples=c)
              for (n, a, b), c in sorted(counts.items())])
    sampled = random.Random(42).sample(test, min(10000, len(test)))
    report = [f"Original train rows: {len(train):,}; unique: {len(train_set):,}",
              f"Original test rows: {len(test):,}; unique: {len(test_set):,}",
              f"Exact ordered train/test overlap: {len(overlap):,} unique expressions",
              f"Test rows occurring in train: {sum(line in train_set for line in test):,}",
              f"Test rows seen in train allowing operand reversal: {sum(unordered(line) in unordered_train for line in test):,}",
              f"Evaluated 10,000-row seed-42 test sample overlap with train: {sum(line in train_set for line in sampled):,}",
              "Original data contain 0- to 99,999-valued operands, NOT exclusively exactly five digits.",
              "The label '5-digit' means the original up-to-five-digit distribution.",
              "New datasets: exactly 6/7 positive operand digits; unique ordered pairs; Python integer targets.",
              "New operand ranges are disjoint from original train/test; no length-OOD overlap.",
              "Evaluation uses the same seed-42 sample of 10,000 original test rows as Train.py."]
    (out / "reports" / "data_audit.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
