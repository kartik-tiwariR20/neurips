# Predicting Dimensional Collapse in Self-Supervised Learning

> Target: **NeurIPS 2026 Workshop — NeurReps** (Symmetry and Geometry in Neural Representations), Proceedings Track (archival, PMLR)

---

## 1. The Research Problem

Self-supervised learning (SSL) models sometimes suffer **dimensional collapse** — the learned embeddings occupy a lower-dimensional subspace than the full embedding dimension, wasting representational capacity.

**Our question:** Given augmentation strength (α), projector width, and dataset structure — can we *predict* the effective rank at convergence, and does the relationship between effective rank and downstream performance *transfer* from vision (where it's been studied) to time-series data?

**Why it matters:**
- Collapse was long treated as a pure failure mode; recent work (Cosentino et al. 2022) shows it can sometimes *help* generalization — but this nuance is established almost entirely in vision-based SSL (SimCLR, BYOL, Barlow Twins, VICReg)
- No existing study tests whether the vision-derived collapse dynamics hold across genuinely different data structures (channel count, sequence length, sample size, class balance)
- If early-training signals can predict final representation quality, that's a real compute-saving diagnostic tool

**What makes this novel:**
1. First systematic test of the augmentation-strength → collapse relationship outside vision, across 5 structurally distinct time-series datasets
2. Tests whether the "collapse can help" effect (Cosentino et al.) is specific to *how* you evaluate — linear probe vs. fine-tuned — not just whether collapse occurred
3. Tests early-epoch predictability (H2) against **both** accuracy regimes, not just one — this distinction turned out to matter enormously: across all four fully-analyzed datasets, early rank significantly predicts final linear-probe accuracy but never fine-tune accuracy (see Section 5c) — a clean, replicating split most single-dataset studies would never surface

---

## 2. Theoretical Framework

### Core Metric: Effective Rank

Given a batch of embeddings `{z_1, ..., z_N}`, compute the covariance matrix:
```
C = (1/N) Σ_i (z_i - z̄)(z_i - z̄)ᵀ
```
Eigendecompose to get `λ_1 ≥ λ_2 ≥ ... ≥ λ_d`, normalize into a distribution `p_i = λ_i / Σ_j λ_j`, then:
```
erank(C) = exp( -Σ_i p_i log p_i )
```
`erank → 1` means near-total collapse; `erank → d` means the embedding space is fully utilized. (Roy & Vetterli, 2007)

### SSL Loss: VICReg

Chosen because its variance term is an explicit, tunable anti-collapse regularizer, making the collapse mechanism directly inspectable:
```
Loss = λ·invariance(z1, z2) + μ·variance(z1, z2) + ν·covariance(z1, z2)
```

### Hypotheses

- **H1 (transfer):** The augmentation-strength → effective-rank relationship established in vision transfers, at least directionally, across structurally different time-series datasets.
- **H2 (predictability):** Effective rank and downstream accuracy at convergence can be predicted from early-training signals (e.g., effective rank at epoch 5), without training to completion.

---

## 3. Experimental Setup

### Shared Pipeline (identical across every dataset — this is what makes cross-dataset comparison fair)

- **Encoder:** small 1D-CNN, fixed capacity, channel-count read automatically from data
- **Projector:** MLP head, width is the swept structural variable (64 vs. 512)
- **Augmentations:** jitter, scaling, time-masking, random-crop-resize, all parametrized by a single strength scalar α ∈ [0, 1]
- **Sweep grid:** 5 α values × 2 projector widths × 3 seeds = 30 runs per dataset
- **Downstream eval:** both linear probe (frozen encoder) and full fine-tune, per run

### Compute

Lightning Studio (CPU and GPU used across different runs, noted per dataset below).

---

## 4. Dataset Inventory

| Dataset | Channels | Seq. Length | Classes | Samples | Domain | Status |
|---|---|---|---|---|---|---|
| **ECG5000** | 1 | 140 | 5 (imbalanced) | 5,000 | Medical/ECG | ✅ Done, fully analyzed |
| **UCI HAR** | 9 | 128 | 6 (balanced) | 10,299 | Motion/wearable | ✅ Done (CPU) |
| **FordA** | 1 | 500 | 2 (balanced) | 4,921 | Mechanical/sensor | ✅ Done, analyzed |
| **Spoken Arabic Digits** | 13 | ~40 (resampled from variable 4–93) | 10 (balanced) | 8,800 | Speech/MFCC | ✅ Done, analyzed |
| **EigenWorms** | 6 | 17,984 | 5 (imbalanced) | 259 | Behavioral genetics | ⏳ In progress — extreme sequence length requires scoped-down sweep (see Section 8) |

This spread deliberately varies channel count (1→13), sequence length (128→17,984), sample size (259→10,299), and class balance — the axes needed to say something precise about *where* the vision-derived relationship holds vs. breaks down, rather than a single pass/fail verdict.

---

## 5. Consolidated Results (datasets with full writeup)

### 5a. H1 — Does augmentation strength predict effective rank?

| Dataset | α=0 erank (w=64) | α=1 erank (w=64) | % collapse | Direction confirmed? |
|---|---|---|---|---|
| ECG5000 | 39.9 | 9.1 | **77%** | ✅ Strong, monotonic |
| UCI HAR | 41.8 | 28.2 | **33%** | ✅ Monotonic, but far weaker magnitude |
| FordA | 38.6 | 12.6 | **67%** | ✅ Strong, monotonic |
| Spoken Arabic Digits | 41.6 | 23.9 | **43%** | ✅ Monotonic |

> [!IMPORTANT]
> The *direction* of H1 replicates cleanly in all four datasets — more augmentation → more collapse, at every projector width tested. A clearer structural pattern is now emerging on *magnitude*: the two univariate datasets (ECG5000, FordA) collapse most (77%, 67%); the two multivariate datasets (HAR at 9 channels, Spoken Arabic Digits at 13 channels) collapse noticeably less (33%, 43%). **Channel count looks like the dominant moderator of collapse magnitude**, more so than sequence length or sample size — though FordA's longer sequence (500 vs. ECG5000's 140) with a similar channel count (1) still shows a meaningfully lower collapse than ECG5000, so sequence length may contribute a secondary effect. EigenWorms (6 channels, extremely long sequence) will help separate these two factors more cleanly.

Projector width effect also replicates in both: wider projector (512) → consistently higher effective rank than narrower (64), at every α tested — matching Garrido et al.'s theory.

### 5b. Effective rank → downstream accuracy

| Dataset | Linear probe r | Linear probe p | Fine-tune r | Fine-tune p |
|---|---|---|---|---|
| ECG5000 | **-0.47** | **0.035** ✅ | +0.29 | 0.213 (ns) |
| UCI HAR | **-0.44** | **0.015** ✅ | -0.23 | 0.221 (ns) |
| FordA | -0.36 | 0.053 (borderline) | -0.03 | 0.894 (ns) |
| Spoken Arabic Digits | **-0.52** | **0.003** ✅ | -0.22 | 0.232 (ns) |

> [!IMPORTANT]
> **This is the headline result so far.** In all four datasets, more collapse (lower effective rank) correlates with *higher or unchanged* linear-probe accuracy — never the opposite — and this relationship **consistently disappears once fine-tuning is allowed** (all four fine-tune p-values are non-significant). Spoken Arabic Digits gives the strongest, most significant linear-probe correlation yet (r=-0.52, p=0.003). The consistency of *which* evaluation regime shows the effect, across four structurally unrelated datasets, is strong evidence this isn't a per-dataset coincidence.
>
> **Anomaly worth flagging:** in Spoken Arabic Digits at α=0.0, width=512 (the highest-effective-rank configuration in the entire sweep, erank≈49), linear probe accuracy collapses to 0.62–0.69 — far below every other configuration (0.88–0.97), and consistent across all 3 seeds (not a fluke). This suggests the erank-accuracy relationship may be **non-monotonic** rather than purely linear: too little collapse (an unregularized, noisy full-rank space) may hurt linear-probe accuracy just as too much collapse does, with some middle ground being optimal. Worth investigating directly before the paper's discussion section is finalized — plot this dataset's full erank-vs-accuracy curve without collapsing across widths to see the shape more clearly.

### 5c. H2 — Early-epoch predictability

| Dataset | vs. Linear probe R² | vs. Linear probe p | vs. Fine-tune R² | vs. Fine-tune p |
|---|---|---|---|---|
| ECG5000 | 0.25 | 0.025 ✅ | 0.07 | 0.273 (ns) |
| UCI HAR | 0.24 | 0.006 ✅ | 0.03 | 0.338 (ns) |
| FordA | 0.17 | 0.024 ✅ | 0.00 | 0.951 (ns) |
| Spoken Arabic Digits | **0.44** | **<0.001** ✅ | 0.03 | 0.377 (ns) |

> [!IMPORTANT]
> **This is now a complete, clean pattern across all four analyzed datasets — 4 for 4.** Early-epoch effective rank significantly predicts final linear-probe accuracy in every single dataset tested (R² ranging 0.17–0.44, all p<0.05), and predicts final fine-tune accuracy in **none** of them (all R² ≤ 0.07, all non-significant). This isn't a fragile result that showed up once — it replicated identically across a medical/ECG dataset, a 9-channel motion dataset, a mechanical sensor dataset, and a 13-channel speech dataset. That consistency, across genuinely different domains and structures, is the strongest single piece of evidence in this study. The practical claim follows directly: **you can estimate final linear-probe representation quality from the first few epochs of training, without running to convergence — but this shortcut does not work if you plan to fine-tune.**

### 5d. Notable anomaly requiring follow-up

The Spoken Arabic Digits α=0/width=512 result (Section 5b) is the most interesting open thread right now — a possible non-monotonic erank-accuracy relationship that the other three datasets' α/width ranges didn't reveal (their least-collapsed configurations weren't as extreme). Worth checking whether ECG5000, HAR, or FordA show any hint of the same pattern at their own highest-erank configurations before concluding this is Spoken-Arabic-Digits-specific.

---

## 6. Pipeline Architecture

```
Raw data (per dataset) → loaders/<name>.py → shared TimeSeriesDataset
        ↓
Augmentation (strength α) → Encoder (channels auto-detected) → Projector (width swept)
        ↓
VICReg loss ← effective_rank() logged every epoch
        ↓
Downstream eval: linear probe AND fine-tune, both logged
        ↓
results/<dataset>/runs/*.json → analyze.py → figures + cross-dataset comparison
```

Adding a new dataset touches exactly two things: one new `loaders/<name>.py`, one line in `loaders/registry.py`. Everything else — model, augmentations, loss, training loop, analysis — is shared, unmodified, identical across all 5 datasets. This is what keeps the cross-dataset comparison methodologically fair.

---

## 7. Known Limitations & Honest Assessment

1. **Only 3 seeds per config** (30 runs/dataset) — correlation estimates have real sampling noise; more seeds would tighten confidence intervals
2. **Accuracy ceiling effects** — ECG5000's fine-tune accuracy is tightly clustered (0.939–0.954), limiting how strong any correlation could ever appear regardless of true effect size
3. **Correlation, not causation** — the erank↔accuracy relationship is observational; no mechanistic claim is made about *why* collapse helps linear-probe accuracy specifically
4. **EigenWorms is a genuine compute outlier** — 17,984-timestep sequences make the default sweep protocol (50 pretrain epochs × 30 eval epochs × 30 runs) impractical even on GPU; a documented, scoped-down protocol is used instead (see Section 8), not a silent shortcut
5. **Two datasets (FordA, Spoken Arabic Digits) have results collected but not yet statistically analyzed** in this document — numbers above are provisional pending that writeup
6. **5 datasets total** — enough for genuine structural spread (channels, length, balance, sample size all vary), but not exhaustive; results should be read as "where we've tested so far," not a universal claim

---

## 8. EigenWorms — Compute Note

Measured directly (CPU): one forward+backward pass at batch=64 takes ~5.5s. With only 259 total samples, that's 4 batches/epoch. At default settings (50 pretrain epochs, 30 eval epochs), a single run is estimated at ~50 minutes — making the full 30-run sweep impractical (~25+ hours) even accounting for GPU speedup (small models like this one often see only modest, not dramatic, GPU acceleration).

**Resolution:** a separate `--eval_epochs` flag was added (decoupled from pretraining `--epochs`) so this dataset's sweep can be scoped down (e.g., 20 pretrain / 15 eval epochs) without altering the shared model architecture that keeps all 5 datasets comparable. This is a documented compute-vs-thoroughness tradeoff specific to this dataset's extreme sequence length, not a change to the method itself.

---

## 9. Next Steps

1. ~~ECG5000 sweep~~ ✅ Done and fully analyzed (both H2 targets)
2. ~~UCI HAR sweep~~ ✅ Done and fully analyzed
3. ~~FordA sweep~~ ✅ Done and analyzed
4. ~~Spoken Arabic Digits sweep~~ ✅ Done and analyzed — strongest H2 result of the study so far
5. **EigenWorms sweep** — scoped-down protocol, in progress; the only remaining dataset. Will help separate channel-count vs. sequence-length effects on collapse magnitude (Section 5a), and serves as the fourth (and most extreme) test of the now 4-for-4 H2 pattern (Section 5c).
6. **Investigate the Spoken Arabic Digits α=0/width=512 anomaly** (Section 5d) — check whether it's an isolated data point or a genuine non-monotonic effect
7. **Cross-dataset comparison** (`analyze.py --cross_dataset`) once EigenWorms is complete — normalized alpha-vs-erank plot, summary correlation table across all datasets
8. **Write paper** — target NeurReps Proceedings submission

---

## 10. Repository Structure

```
neurips/
├── README.md                  (this file)
├── requirements.txt
├── train.py / sweep.py / analyze.py
├── model.py / losses.py / effective_rank.py / evaluate.py / augmentations.py
├── loaders/
│   ├── registry.py             (one entry per dataset)
│   ├── common.py                (shared normalize/split logic)
│   └── ecg5000.py / har.py / forda.py / eigenworms.py / spokenarabicdigits.py
├── data/raw/<dataset_name>/    (raw files, per dataset)
└── results/<dataset_name>/
    ├── runs/                    (per-run JSON logs)
    └── figures/                 (per-dataset plots + results table)
```
