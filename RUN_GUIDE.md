# How to run this — step by step

Everything here has already been smoke-tested end-to-end (3-epoch run, tiny 2-config
sweep, and the analysis script) so you know it works before committing real compute.

## 0. Setup

```bash
cd ssl_collapse
pip install -r requirements.txt
```

Data is already in `data/raw/ECG5000_TRAIN.txt` and `data/raw/ECG5000_TEST.txt`
(the two files pulled from your upload — the `.ts` and `.arff` versions aren't
needed, the loader uses the plain `.txt` format).

## 1. Sanity-check each piece individually (optional but recommended)

```bash
python3 dataset.py         # confirms data loads: 5000 series, splits print out
python3 augmentations.py   # confirms augmentation strength scales with alpha
python3 model.py           # confirms encoder/projector shapes
python3 effective_rank.py  # confirms metric: ~64 for full-rank, ~1 for collapsed
python3 losses.py          # confirms VICReg loss computes
```

## 2. Run one quick training run (~1-2 min on CPU)

```bash
python3 train.py --alpha 0.5 --projector_width 64 --epochs 20
```

This writes `runs/alpha0.50_proj64_seed0.json` containing the config, the
per-epoch effective-rank trajectory, and downstream accuracy (linear probe +
fine-tune). Confirm it runs and the printed numbers look sane before the full sweep.

## 3. Run the full sweep

Default grid is 5 alphas × 2 projector widths × 2 seeds = **20 runs**, 50 epochs
each. On CPU, expect roughly **60-90 minutes total** (each run is ~2-4 min).
If you have a GPU, it'll be much faster — the code already auto-detects CUDA.

```bash
python3 sweep.py --epochs 50 --out_dir runs
```

Want 30 runs instead of 20? Add a third seed:

```python
# edit sweep.py, change:
SEEDS = [0, 1, 2]
```

Each run's JSON lands in `runs/`. If a run crashes partway, you can safely
re-run just that one config with `train.py` directly — sweep.py doesn't
need to be re-run from scratch since each run's file is independent.

## 4. Analyze results

```bash
python3 analyze.py --run_dir runs --fig_dir figures
```

This produces, in `figures/`:
- `results_table.csv` — every run's config + metrics, one row each (this is
  your raw results table for the paper's experiments section)
- `alpha_vs_erank.png` — the core H1 plot: does augmentation strength predict
  final effective rank, split by projector width?
- `erank_vs_accuracy.png` — does effective rank correlate with downstream
  accuracy (both linear-probe and fine-tune), with Pearson r and p-value?
- `early_prediction.png` — the H2 test: does the effective rank at epoch 5
  (early training) predict the FINAL downstream accuracy? Includes R² and
  p-value from the regression fit.

## 5. What to look for, and what it means for your paper

- **If `alpha_vs_erank.png` shows a clear downward trend** (higher alpha →
  lower effective rank), that's your first confirmation that the vision-derived
  relationship (Jing et al.) transfers to time-series — supports H1.
- **If `erank_vs_accuracy.png` shows a non-monotonic relationship** (e.g. an
  inverted-U, best accuracy at moderate collapse, not zero collapse) — that's
  the "collapse can help" effect (Cosentino et al.) showing up in your modality.
  This would be a genuinely interesting result.
- **If `early_prediction.png` shows a strong R² (say > 0.5) with low p-value**,
  that's a real H2 result: you can predict downstream performance early,
  without training to convergence — the practical, compute-saving contribution.
- **If any of these come back flat or noisy** — that's not a failed experiment,
  it's a finding. A paper section explaining *where* the vision intuition
  breaks down for time-series is legitimate and often more interesting than a
  clean confirmation. Don't discard the sweep if results are messy; write up
  what you found honestly.

## 6. Extending to a second/third dataset (for the "dataset structure" comparison)

Once ECG5000 sweep is done, repeat with a second dataset (e.g. UCI HAR or
BasicMotions) to test whether the relationship holds across different
channel counts / class balance / series lengths — this is the cross-structure
comparison your paper needs beyond just "it works on one dataset."

To do this: write a new loader function analogous to `load_ecg5000()` in
`dataset.py` for the new dataset's format, point `train.py`'s `data_dir`
argument at it, and re-run `sweep.py` with a different `--out_dir` (e.g.
`runs_har/`) so results don't overwrite ECG5000's. `analyze.py` can then be
pointed at each `runs_*` directory separately, or combined by loading both
CSVs and adding a `dataset` column for a cross-dataset comparison plot.

Send me the new dataset's files when you're ready and I'll write that loader
the same way I did for ECG5000.
