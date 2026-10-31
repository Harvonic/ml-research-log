# Local model playground

A disposable localhost demo: **original baseline on the left, original NULL
checkpoint on the right**. A shared input runs both models and shows their
generated answers, correctness, token counts, and approximate generation times.

## Run

From the repository root:

```powershell
python local_demo/server.py --open
```

Or double-click `local_demo/start.bat` on Windows. The browser opens at
**http://127.0.0.1:8765** after both checkpoints load. Keep the terminal open;
press **Ctrl+C** there to stop the server. If the port is busy:

```powershell
python local_demo/server.py --port 8766 --open
```

Uses the project's existing Python/PyTorch installation and Python's built-in
HTTP server. CUDA is selected when available, otherwise CPU. No frontend build
or additional Python dependencies are required.

## Try it

- Enter `12345 + 67890` and click **Compare**, or press Enter.
- The random buttons fill an exactly 5-, 6-, or 7-digit example; press Compare
  to run it. Inputs support nonnegative operands with up to 12 digits each.
- Four NULL tokens are inserted into the NULL model's prompt automatically.
  The model does not choose their count or generate NULLs as its answer.
- Predictions come from the actual checkpoints with the original greedy
  decoding rules. The Python-computed sum is used only to judge correctness.
- Timing is a single-request measurement, after a small startup warm-up.
  Models run sequentially, so these values are not a benchmark.
- Recent comparisons stay in page memory only and clear on reload.

## Remove it

Delete **only `local_demo/`**. All demo code and assets live here. It reads
`../src/Model.py`, `../src/ExperimentUtils.py`, and the original
`../results/baseline.pt` / `../results/null.pt` without modifying them. There
are no copied checkpoints, saved predictions, or changes to experiment files.
Keep the folder inside this repository so those relative paths resolve.
