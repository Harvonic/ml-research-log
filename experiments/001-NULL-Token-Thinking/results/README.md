# Results

Start with the experiment summaries:

- [Generalization and NULL ablations](generalization/summary.txt)
- [FLOP-matched comparison](flop_matched/summary.txt)

```text
results/
  baseline.pt                 Original baseline checkpoint
  null.pt                     Original NULL checkpoint
  training_log.txt            Original training log
  plots/                      Original experiment plots
  generalization/
    summary.txt
    plots/                    Generalization and ablation figures
    metrics/                  Evaluation, carries, dataset and operand counts
    logs/                     Evaluation console log
    reports/                  Data audit, validation and protected-file hashes
  flop_matched/
    summary.txt
    baseline_matched.pt       New baseline checkpoint
    plots/                    Compute comparison and training efficiency
    metrics/                  Evaluation, comparison, budget and training CSVs
    logs/                     Training and console logs
    reports/                  FLOP method and plotting notes
```

Original files retain their paths for compatibility with the unchanged original
training and plotting scripts. Each extension script reads and writes the organized
experiment folders above. Checkpoints and raw numeric results are preserved.

For run commands, assumptions, and conclusions, see [EXPERIMENTS.md](../EXPERIMENTS.md).
