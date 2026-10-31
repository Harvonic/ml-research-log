
import random
from pathlib import Path

N = 1_000_000
LIMIT = 99_999
# Train seed
random.seed(1337)

# Test seed
# random.seed(67)

data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(exist_ok=True)

with open(data_dir / "train.txt", "w") as f:
    for _ in range(N):
        op = "+"

        a = random.randint(0, LIMIT)
        b = random.randint(0, LIMIT)

        if op == "+":
            answer = a + b
        # elif op == "-":
        #     answer = a - b
        # elif op == "*":
        #     answer = a * b
        # else:
        #     # Generate division with an integer answer
        #     b = random.choice([-1, 1]) * random.randint(1, 10_000)
        #     answer = random.randint(-100_000, 100_000)
        #     a = b * answer

        f.write(f"{a} {op} {b} = {answer}\n")

print(f"Generated {N:,} problems in {data_dir}")
