# Predicting Dimensional Collapse in Self-Supervised Learning

**Research question:**
> Given augmentation strength α, projector width, and dataset structure, can we predict the effective rank at convergence — and does the relationship between effective rank and downstream performance transfer from vision to [chosen modality]?

**Target venue:** NeurIPS 2026 Workshop — *NeurReps* (Symmetry and Geometry in Neural Representations), Proceedings Track (archival, published via PMLR)

---

## 1. Motivation

Dimensional collapse — where learned SSL embeddings occupy a lower-dimensional subspace than the full embedding dimension — was long treated as a pure failure mode. Recent work has shown this is not always true: collapse can *help* generalization under certain conditions (strong augmentation, wide projectors), and *hurt* it under others. This nuance has been established almost entirely in vision-based contrastive/non-contrastive SSL (SimCLR, BYOL, Barlow Twins, VICReg). Two gaps remain open:

1. **Cross-modal transfer**: does the vision-derived relationship between augmentation strength, projector width, and effective rank hold in other modalities (time-series, audio, tabular, graph)?
2. **Predictability**: can the *final* effective rank and downstream performance be predicted from *early-training* signals, without running training to convergence?

This project targets both gaps with one unified experimental design.

## 2. Background Reading (in order)

| # | Paper | Why it matters |
|---|-------|-----------------|
| 1 | Chen et al. 2020 — SimCLR | Baseline contrastive framework, InfoNCE loss |
| 2 | Grill et al. 2020 — BYOL | Non-contrastive SSL, collapse without negatives |
| 3 | Zbontar et al. 2021 — Barlow Twins | Covariance-based anti-collapse regularization |
| 4 | Bardes et al. 2022 — VICReg | Explicit variance term against eigenvalue collapse |
| 5 | Jing et al. 2022 (ICLR) — *Understanding Dimensional Collapse in Contrastive SSL* | Foundational theory: strong augmentation + implicit regularization cause low-rank covariance |
| 6 | Cosentino et al. 2022 | First empirical evidence that collapse can *improve* generalization |
| 7 | ICLR 2023 — loss landscape of SSL | Analytic/closed-form reconciliation of "collapse helps vs. hurts" |
| 8 | Garrido et al. 2023 — duality of contrastive/non-contrastive SSL | Projector vs. encoder collapse distinction |
| 9 | He & Ozay 2022 | Collapsed vs. whitened feature spaces |
| 10 | OrthoReg (NeurIPS 2024) | Weight-space collapse, beyond just output features |
| 11 | AdaDim (2025) | Adaptive dimensionality control during training — closest prior art to this proposal |
| 12 | Roy & Vetterli 2007 | Definition of effective rank (the core metric used throughout) |

## 3. Core Metric: Effective Rank

Given a batch of embeddings `{z_1, ..., z_N}`, compute the covariance matrix:

```
C = (1/N) Σ_i (z_i - z̄)(z_i - z̄)ᵀ
```

Eigendecompose `C` to get eigenvalues `λ_1 ≥ λ_2 ≥ ... ≥ λ_d`. Normalize into a probability distribution:

```
p_i = λ_i / Σ_j λ_j
```

Effective rank (Roy & Vetterli, entropy-based):

```
erank(C) = exp( -Σ_i p_i log p_i )
```

This gives a single scalar per checkpoint. `erank → 1` means near-total collapse (one dominant direction); `erank → d` means the embedding space is fully utilized.

## 4. Hypotheses

- **H1 (transfer):** The augmentation-strength → effective-rank relationship established in vision transfers *quantitatively* (not just qualitatively) to the chosen modality.
- **H2 (predictability):** The effective rank at convergence — and downstream accuracy — can be predicted from early-training statistics (e.g., the effective-rank trajectory over the first few epochs), avoiding the need to train to completion.

## 5. Experimental Design

**Modality choice:** time-series (most tractable — single GPU feasible, existing SSL frameworks like TS2Vec / TS-TCC to adapt).

**Independent variables (swept):**
- Augmentation strength α ∈ [0, 1] (jitter / masking / cropping intensity)
- Projector width (encoder-output-dim multiplier)
- Batch size

**Dependent variables (logged every epoch):**
- Effective rank of the embedding covariance matrix
- Linear-probe downstream accuracy
- Fine-tuned downstream accuracy (checks whether "optimal collapse" holds under both eval regimes — a common gap in prior work)

**Core experiment:** ~20–30 runs sweeping α and projector width on a benchmark such as the UCR Archive or HAR dataset. Log effective-rank trajectories, correlate final effective rank with downstream accuracy, and fit a lightweight predictive model (e.g., regression) from early-epoch rank trajectory → final downstream accuracy. Compare accuracy-per-compute against full-training baselines.

**Optional theoretical extension:** adapt the ICLR 2023 loss-landscape analysis (Hessian eigenspectrum near the collapse fixed point) to the augmentation structure of the chosen modality.

## 6. Math Toolkit Required

- Linear algebra: SVD/eigendecomposition of covariance matrices, rank, condition number
- Multivariable calculus & perturbation theory: for loss-landscape / Hessian analysis
- Information theory (optional): mutual information estimators, InfoNCE as an MI bound, information-bottleneck framing of "how much collapse is optimal"
- Basic statistics: correlation, regression, significance testing across the α × projector-width sweep

## 7. Proposed Paper Structure

1. Introduction — motivate cross-modal gap in dimensional collapse literature
2. Background — Jing et al. theory + Cosentino et al. contradiction
3. Method — predictive framework + cross-modal experimental design
4. Theory (optional) — loss-landscape extension to new modality
5. Experiments — effective-rank-vs-accuracy sweep, early-prediction results
6. Discussion — where vision-derived intuition breaks down (often the most interesting section)

## 8. Timeline (targeting NeurReps Proceedings, NeurIPS 2026)

| Phase | Duration | Output |
|---|---|---|
| Literature review + math foundations | 2 weeks | Annotated bibliography, derivations notebook |
| Codebase setup + baseline SSL pipeline | 1–2 weeks | Working SSL training loop for chosen modality |
| Core sweep experiments (α × projector width) | 2–3 weeks | Effective-rank & accuracy logs across ~20–30 runs |
| Predictive model + analysis | 1–2 weeks | Early-prediction results, accuracy-per-compute comparison |
| Writing + figures | 1–2 weeks | Draft paper |
| Buffer / revisions | 1 week | Final submission |

*(Adjust against the actual NeurReps 2026 submission deadline once confirmed on neurreps.org — 2025's was August 29.)*

## 9. Submission Fit — Honest Assessment

**Why NeurReps Proceedings is a good fit:**
- Its explicit scope includes representational geometry, dynamics of neural representations, and geometric approaches to interpretability — effective rank and loss-landscape analysis sit squarely inside this.
- The Proceedings track is archival (PMLR), so a solid version of this paper yields a real citable publication, not just a poster.

**Realistic chances, and what determines them:**
- The novelty claim is defensible (cross-modal transfer + early prediction) but *narrow* — success depends on **execution quality**, not idea novelty alone. Workshop reviewers will want to see: (a) a clean, reproducible sweep with enough runs to support a correlation claim, (b) an honest negative-result discussion if transfer *doesn't* hold cleanly, and (c) a prediction result that beats a naive baseline (e.g., "just use α as the predictor").
- Biggest risk: running out of time before the deadline with only partial experiments. Scope the modality and dataset tightly (one modality, one or two benchmarks) rather than trying to cover several.
- A negative or partial result (e.g., "the vision relationship holds for accuracy but not for effective-rank magnitude") is still a publishable, interesting workshop paper — don't treat a clean confirmation as the only success condition.
- Workshops are lower-stakes and more welcoming to in-progress or narrowly-scoped work than the main conference, which improves realistic odds relative to a NeurIPS main-track submission.

**Bottom line:** plausible and worth pursuing, contingent on tight scoping and disciplined timeline execution — not a novelty problem, an execution problem.

## 10. Open Items to Verify Before Committing

- [ ] Confirm NeurReps 2026 submission deadline and page limit at neurreps.org
- [ ] Check OpenReview / arXiv for any very recent (last 1–2 months) papers scooping the cross-modal-collapse angle
- [ ] Decide final modality + benchmark dataset
- [ ] Confirm compute budget (single GPU vs. more) against the planned sweep size
