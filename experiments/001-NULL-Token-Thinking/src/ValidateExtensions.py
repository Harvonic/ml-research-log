"""Check extension batching against the untouched original helper functions."""
import ast
import csv
import random
import re

import torch

from ExperimentUtils import SRC, ROOT, DEVICE, encode, load_model, read_lines, sha256, write_csv
from EvaluateGeneralization import evaluate


def main():
    tree = ast.parse((SRC / "Train.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in ("make_batch", "predict")]
    namespace = dict(torch=torch, DEVICE=DEVICE, MAX_ANSWER_TOKENS=24)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SRC / "Train.py"), "exec"), namespace)
    examples = ["0 + 0 = 0", "12 + 345 = 357", "99999 + 99999 = 199998"]
    torch.set_num_threads(4)
    checks = []
    for model_name, count in (("baseline", 0), ("null", 4)):
        model, vocab = load_model(ROOT / "results" / f"{model_name}.pt")
        expected_x, expected_y = namespace["make_batch"](examples, vocab, count)
        x, y, _ = encode(examples, vocab, count)
        assert torch.equal(x.to(DEVICE), expected_x) and torch.equal(y.to(DEVICE), expected_y)
        cases = examples + random.Random(42).sample(read_lines(SRC / "data/test.txt"), 12)
        expected_hits = sum(namespace["predict"](model, line.split(" = ")[0], vocab, count) == line.split(" = ")[1]
                            for line in cases)
        metric, _ = evaluate(model, vocab, cases, count, "<NULL>", 8)
        assert metric["correct"] == expected_hits
        checks.append(f"{model_name}: original and extension inputs/targets identical; batched vs original scalar exact-match counts agree on {len(cases)} cases.")
    for digits in (6, 7):
        lines = read_lines(SRC / "data/generalization" / f"addition_{digits}_digit.txt")
        assert len(lines) == len(set(lines)) == 10000
        for line in lines:
            problem, answer = line.split(" = ")
            a, b = map(int, problem.split(" + "))
            assert 10 ** (digits - 1) <= a < 10 ** digits
            assert 10 ** (digits - 1) <= b < 10 ** digits
            assert a + b == int(answer)
        checks.append(f"{digits}-digit data: 10,000 unique valid sums and exact operand lengths verified independently.")
    protected = [SRC / name for name in ("Model.py", "Train.py", "GenerateData.py", "PlotResults.py", "vocab.json")]
    protected += [SRC / "data" / name for name in ("train.txt", "test.txt")]
    protected += [ROOT / "results" / name for name in ("baseline.pt", "null.pt", "training_log.txt")]
    manifest = ROOT / "results/generalization/reports/protected_files.csv"
    hashes = [dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)) for path in protected]
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as f:
            assert list(csv.DictReader(f)) == hashes, "A protected file changed"
        checks.append("All ten protected source/data/checkpoint/log hashes remain unchanged.")
    else:
        write_csv(manifest, hashes)
        checks.append("Recorded hashes of protected source/data/checkpoints/log for future reproducibility checks.")
    metrics = ROOT / "results/flop_matched/metrics/training_metrics.csv"
    if metrics.exists():
        raw = (ROOT / "results/training_log.txt").read_bytes()
        text = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        baseline_text = text.split("null:")[0]
        reference = {}
        for line in baseline_text.splitlines():
            match = re.search(r"epoch (\d+) \| step (\d+) \| avg loss ([\d.]+)", line)
            if match:
                reference[int(match[1]), int(match[2])] = float(match[3])
            match = re.search(r"epoch (\d+) finished \| avg loss ([\d.]+)", line)
            if match:
                reference[int(match[1]), 15625] = float(match[2])
        with metrics.open(encoding="utf-8", newline="") as f:
            new_rows = list(csv.DictReader(f))
        differences = [abs(float(row["epoch_mean_loss"]) - reference[key])
                       for row in new_rows if (key := (int(row["epoch"]), int(row["epoch_step"]))) in reference]
        assert differences and max(differences) <= 0.000051
        checks.append(f"Fresh training replay agrees with {len(differences)} original baseline loss log entries within their four-decimal precision.")
    matched = ROOT / "results/flop_matched/baseline_matched.pt"
    if matched.exists():
        checkpoint = torch.load(matched, map_location="cpu", weights_only=True)
        with (matched.parent / "metrics" / "compute_budget.csv").open(encoding="utf-8", newline="") as f:
            expected = list(csv.DictReader(f))[-1]
        assert checkpoint["training_steps"] == int(expected["steps"])
        assert checkpoint["examples_processed"] == int(expected["examples"])
        assert checkpoint["estimated_training_flops"] == int(expected["estimated_training_flops"])
        checks.append("Saved matched checkpoint step/example/FLOP metadata agree with the reconstructed target schedule.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    (manifest.parent / "validation.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print("\n".join(checks))


if __name__ == "__main__":
    main()
