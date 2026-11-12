# Soft Exclusive Self-Attention sweep

Edit `LAM_VALUES` and `SEEDS` at the top of `src/Train.py`, then run from the project root:

```powershell
python src/Train.py
```

You can also run `python Train.py` from `src/`. The tokenizer and results paths are relative to the script, so both commands use the same files. Dependencies are PyTorch, NumPy, datasets, tokenizers, and Matplotlib.

The default lists run 25 independent training jobs: all five seeds for lambda 0, then all five for lambda 0.5, and so on. Data is loaded and tokenized once. Each job creates a fresh model and optimizer; matching seeds use matching initial weights and training windows across lambdas. CUDA operations can still introduce nondeterminism.

## Output layout

```text
results/
  xsa-lam=1/
    resultsJson/seed_5/metrics.json
    visuals/seed_5/metrics.png
    checkpoints/seed_5/model.pt
```

Each lambda and seed gets the same layout. Checkpoints contain the **final** model weights; `best_step` records the best observed validation loss but does not imply a saved best checkpoint.

Existing seed folders stop the sweep before data loading to prevent accidental overwrites. Move previous outputs or change `RESULTS_DIR` to start another sweep with the same combinations. The comparison files summarize completed runs from the current invocation only and update after each run; they do not combine historical experiments with different settings.

## Metrics

Each run's JSON contains settings, a summary, and one history entry per evaluation interval (plus the final step if needed). The default 5,000 steps and 250-step interval produce 20 history entries.

- `train_loss`: average next-token cross-entropy over training batches since the previous evaluation.
- `val_loss`: average cross-entropy over `EVAL_BATCHES` sampled validation windows. The same seed uses the same windows at each evaluation and across lambdas, independently of training sampling.
- `val_accuracy`: next-token accuracy as a fraction from 0 to 1, on those validation windows. Plots display percentages.
- `final_val_perplexity`: exponential of the final validation loss; `null` if too large to represent safely.
- `best_val_loss` and `best_step`: best sampled validation result observed during that run.
- `elapsed_seconds`: time since the training loop began, including evaluation and output work. The final summary also includes checkpoint and plot saving. Dataset preparation and model setup are excluded.
- `training_seconds` and `training_tokens_per_second`: training loop time and throughput, excluding validation and output work. CUDA is synchronized at timing boundaries.

Per-seed plots show loss, validation accuracy, and training throughput. The comparison chart shows final validation loss, final accuracy, and average training throughput as means with sample standard deviations across completed seeds. Standard deviation is `null` with only one completed seed.

JSON is replaced atomically after each evaluation. If interrupted, the most recent saved measurements remain with status `running`; automatic checkpoint resume is not implemented.
