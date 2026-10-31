# Extension experiments

The original scripts, data, checkpoints, logs, and `results/plots/` are preserved.
New outputs are grouped by experiment:

| Location | Contents |
| --- | --- |
| `src/data/generalization/` | Separate deterministic 6/7-digit datasets |
| `results/generalization/` | Summary plus `metrics/`, `logs/`, `reports/`, and `plots/` |
| `results/generalization/plots/` | Generalization, NULL count, and embedding ablation PNGs |
| `results/flop_matched/` | New checkpoint and summary plus `metrics/`, `logs/`, `reports/`, and `plots/` |
| `results/flop_matched/plots/` | Compute-matched accuracy and loss versus FLOPs PNGs |

Completed results (10,000 cases per distribution, seed 42):

| Model | Original up-to-5-digit accuracy | 6-digit accuracy | 7-digit accuracy | Original answer loss |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 95.48% | 0% | 0% | 0.032913 |
| Original NULL × 4 | 98.41% | 0% | 0% | 0.012838 |
| FLOP-matched baseline | 96.57% | 0% | 0% | 0.025909 |

The matched baseline trained for 93,576 steps / 5,988,864 examples, using an
estimated 71.659968 trillion FLOPs versus the NULL target of 71.660160 trillion
(-0.000268% mismatch). NULL keeps a 1.84 percentage-point accuracy advantage
at matched estimated training compute in this single seed. None of the models
shows exact-answer length generalization on these datasets.

Every tested NULL count other than four scored 0%; replacing four NULL tokens
with UNK scored 1.52% on the original distribution. These input distribution
shifts do not establish learned reasoning or separate positional effects.
Original-distribution inference costs approximately 26.61 million FLOPs/problem
for NULL versus 21.60 million for either baseline, using full-prefix generation.

The complete original train/test audit found 97 exact overlapping expressions,
but none in the evaluated 10,000-row test sample. All ten protected source/data/
checkpoint/log hashes stayed unchanged, and the fresh training run reproduced
160 original baseline loss log entries within their four-decimal precision.
See [generalization summary](results/generalization/summary.txt) and
[compute-matched summary](results/flop_matched/summary.txt) for the full caveats.

Run from the repository root with Python, PyTorch, NumPy, and Matplotlib installed:

```powershell
python src/GenerateGeneralization.py
python src/ValidateExtensions.py
python src/EvaluateGeneralization.py
python src/TrainFlopMatched.py --estimate-only
python src/PlotExperiments.py
python src/TrainFlopMatched.py
python src/EvaluateGeneralization.py --matched
python src/PlotExperiments.py
```

The generator refuses to replace differing generated datasets. The trainer refuses
to overwrite an existing matched checkpoint. Evaluation/plot commands refresh only
their extension outputs. Training is from scratch, using the original seed,
initialization, data shuffling, batch size, optimizer defaults, clipping and learning
rate. Pre-encoding preserves batch contents and targets while reducing CPU overhead.
No original training script is imported or executed by the new trainer.

Evaluation uses each checkpoint's vocabulary and EOS ID. Greedy answer accuracy is
primary; token-weighted answer cross-entropy includes EOS. The original test set has
one million examples; the original seed-42 sample of 10,000 is reused for comparable
evaluation. Original operands range from zero to 99,999 and are not exclusively
five digits. See `results/generalization/reports/data_audit.txt` for original data overlap.

`results/flop_matched/reports/flop_method.txt` documents the analytical estimator. It counts dense projections,
full-square attention, feedforward and output matmuls with actual batch padding,
then approximates forward plus backward as three times forward. It does not measure
hardware operations, wall time, or energy. The target is reconstructed from the
current original code/data and five training epochs; the closest whole baseline
optimizer step is selected. Inference estimates count full-prefix generation through
EOS, without a KV cache. Training matching leaves NULL's extra inference cost.

Other NULL counts and UNK replacement are distribution shifts. Removal also changes
positions. Their results cannot establish reasoning or fully separate embedding,
sequence-position and computation effects. This is a single-seed experiment.
