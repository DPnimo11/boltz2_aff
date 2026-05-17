# Project Context

This repository builds **per-target** models that predict ULVSH ligand affinity
from Boltz-2 affinity-module embeddings, and benchmarks them against Boltz-2's
own scalar outputs (B2-A binding affinity, B2-C binding probability).

The methodology baseline comes from Bret, Sindt, Rognan (J. Chem. Inf. Model.
2026, 66, 1511-1521), `papers/assessing-boltz-2-performance-...pdf`. That paper
reports per-target ROC AUC of raw Boltz-2 outputs against ULVSH's
active/inactive labels and uses AUC > 0.65 as "acceptable" with a median
across 10 targets of 0.763.

## Repository Layout

- Outer directory `boltz2_aff/` is the **repository root** (pyproject, data,
  runs, scripts, papers, this file).
- Inner `boltz2_aff/` is the **importable Python package** loaded by
  `python -m boltz2_aff.pipeline`. Modules inside:
  - `data.py` — ULVSH label loader and `p_affinity` derivation.
  - `features.py` — embedding/scalar discovery, `EMBEDDING_KEY_CHOICES`.
  - `modeling.py` — `train_classifier`, `train_regressor`,
    `_screening_auc`, `boltz_baseline_metrics`.
  - `pipeline.py` — CLI driver.
- `scripts/sweep_embedding_keys.py` — embedding-component sweep harness.

## Current Data Layout

- `data/ULVSH/<target>/raw/vitro.tsv` contains labels: ligand ID,
  target-specific affinity/activity measurement, and active/inactive status.
- `data/ULVSH/<target>/raw/scores.tsv` contains the original ULVSH docking and
  physics score features.
- `data/Boltz-2/<target>/<variant>/output/<ligand>/affinity_<ligand>.json`
  contains scalar Boltz affinity predictions (used for the raw-Boltz baseline).
- Future Boltz embedding exports should appear as
  `affinity_embeddings_<ligand>.npz` next to the scalar affinity JSON.
- The currently available extracted embedding set is
  `targets/ROCK1/affinity_embeddings_*.npz` (68 of 69 ROCK1 ligands; `mol_44`
  is missing).

## Embedding Provenance

Each `affinity_embeddings_<ligand>.npz` exported by the `../boltz` fork
contains four arrays:

- `affinity_embedding_pair_mean1` — 128-dim pooled receptor-ligand interface
  pair representation (ensemble member 1) immediately before the scalar
  affinity heads.
- `affinity_embedding_pair_mean2` — same, ensemble member 2.
- `affinity_embedding_head1` — 384-dim representation after the final
  affinity MLP (ensemble member 1) before the scalar prediction heads.
- `affinity_embedding_head2` — same, ensemble member 2.

`boltz2_aff/features.py` flattens these into numeric columns prefixed `emb_`.
Use `--embedding-keys pair_mean1 [pair_mean2 head1 head2]` to restrict.

## Modeling Defaults

- Classification uses the ULVSH `Active` column and can include rows whose
  affinity/activity measurement is nonnumeric or percent-style (`<40%`).
- Regression uses only uncensored numeric affinity measurements (`Ki`, `EC50`,
  `IC50`, `Kd`, or provided `pki`) and trains on
  `p_affinity = 6 - log10(value_uM)` so larger values mean stronger binding.
- Cross-validation groups rows by `target::ligand_id` so multiple Boltz
  variants for the same ligand cannot leak across folds.
- The default linear baselines are deliberately conservative:
  L2-regularized logistic regression for classification, ridge regression
  (`RidgeCV` over `np.logspace(-4, 4, 33)`) for `p_affinity`. The pipeline
  always emits both `classifier.joblib` and `regressor.joblib`.

## Feature Sets

- `embeddings` (default) — flattened `affinity_embeddings_*.npz` arrays only.
- `boltz` — scalar Boltz JSON fields (`boltz_affinity_pred_value`,
  `boltz_affinity_probability_binary`, plus their ensemble suffixes).
- `ulvsh_scores` — original ULVSH docking/physics columns (~25-28 columns).
- `combined` — concatenation of the three above. For ROCK1 this is roughly
  1024 + 6 + 25 = 1055 features.

Regardless of feature set, whenever Boltz affinity JSONs exist they are also
merged into `dataset.csv` as metadata columns so the raw-Boltz baseline AUC
can be computed alongside the model.

## Metrics Reported

For each task, `manifest.json` and `models/metrics_*.json` contain
group-grouped cross-validated metrics. Key fields:

- Classification: `cv_roc_auc`, `cv_average_precision`, `cv_accuracy`,
  `cv_balanced_accuracy`, `cv_log_loss`.
- Regression: `cv_rmse_p_affinity`, `cv_mae_p_affinity`, `cv_r2`,
  `cv_pearson_r`, `cv_spearman_r`, and `cv_roc_auc` — a screening AUC where
  each fold trains the regressor on the in-fold numeric-affinity rows and
  predicts `p_affinity` for *all* test rows (including censored inactives),
  then computes AUC vs `active_bool`. This mirrors how the paper evaluates
  Boltz-2's outputs.
- `boltz_baseline` — per-target raw Boltz AUC (B2-A and B2-C) against
  `active_bool` plus Pearson/Spearman against `p_affinity`. No model fit.

## Main Commands

Single ROCK1 run (default = embeddings, all four components):

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 --out-dir runs/rock1_embeddings
```

Restrict embedding components:

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 --embedding-keys pair_mean1 --out-dir runs/rock1_pm1
```

Compare against scalar Boltz or ULVSH scores:

```powershell
python -m boltz2_aff.pipeline --feature-set boltz --out-dir runs/boltz_scalar
python -m boltz2_aff.pipeline --feature-set ulvsh_scores --out-dir runs/ulvsh_scores
python -m boltz2_aff.pipeline --feature-set combined --out-dir runs/combined
```

Embedding-component sweep:

```powershell
python scripts/sweep_embedding_keys.py --target ROCK1 --out-root runs/rock1_sweep
python scripts/sweep_embedding_keys.py --target ROCK1 --out-root runs/rock1_sweep_combined --feature-set combined
```

## Multi-Target Sweep Findings (2026-05-17)

All 10 targets now have embeddings under `targets/`. Full per-target sweep
(9 embedding combos × `embeddings` and `combined` feature sets) is in
`runs/sweep_embeddings/` and `runs/sweep_combined/` (`summary_by_target.json`,
`medians.json`).

**ROCK1 was an outlier — the embedding models do NOT clearly beat raw Boltz
across targets.**

Classification (combined feature set, honest *fixed* `pair_mean1` choice, not
post-hoc best-combo) vs the better of raw Boltz B2-A/B2-C per target:

- Median embedding cls AUC 0.753 vs median raw Boltz 0.768 — a slight loss.
- `pair_mean1` beats raw Boltz on only **5/10** targets.
- Big embedding wins: CASR (0.824 vs 0.645), DRD3 (0.943 vs 0.846),
  ROCK1 (0.908 vs 0.854). Big losses: ADRA2B (0.250 vs 0.917, only n=13),
  MTR1A (0.442 vs 0.597).
- The earlier "8/10 win" figure was post-hoc best-combo selection on the same
  CV (optimistic bias); the fixed-choice number above is the honest one.

No universal best embedding component: ROCK1→pair_mean, CASR/CNR2/SC6A4→head1,
ADRA2B→head2. The ROCK1-only "pair_mean always wins" conclusion does **not**
generalize.

Regression is effectively broken outside ROCK1:

- 4/10 targets (CNR1, DRD3, SC6A4, SGMR2) have **zero** uncensored numeric
  affinity rows — regression is skipped entirely.
- Of the 6 that fit, only ROCK1 (n=27) gives a strong positive Pearson
  (~0.80). CASR (n=148) is weakly positive (~0.50); CNR2/MTR1A/DRD4/ADRA2B
  are near-zero or negative. Median regression Pearson across targets is
  negative for almost every combo.

Bottom line: the user's original intuition holds — embeddings do not beat the
native Boltz-2 affinity predictor in aggregate. ROCK1 was a lucky target.

## Caveats

- Per-target n is tiny for several targets (ADRA2B clsN=13, DRD3 clsN=32,
  SC6A4 clsN=33, MTR1A clsN=36). AUC at n≈13 is essentially noise — ADRA2B's
  0.250 should not be over-interpreted.
- Picking the best combo per target using the same CV that scores it is
  optimistically biased. Report a single fixed combo for honest comparison,
  or add nested CV / a held-out split before claiming wins.
- ULVSH does not provide numeric IC50/Ki for inactives (censored or
  percent-style), so the regression subset is all-active. Hence the screening
  AUC implementation that predicts on inactives despite training on actives.
- Regression is only meaningfully trainable for CASR (148), DRD4 (74), and
  ROCK1 (27); elsewhere there is too little numeric affinity data.

## Possible Next Steps

- Add nested CV or a held-out test split so per-target combo selection is not
  optimistically biased; re-evaluate the classification "wins" honestly.
- Focus regression effort on CASR/DRD4/ROCK1 only — the rest lack numeric
  affinity data. Consider dropping regression from the headline entirely and
  framing the project as binary classification (as the paper does).
- Try PLS regression or PCA→ridge to handle p≫n for the `head` components
  rather than discarding them.
- Train the regressor on `p_affinity − boltz_affinity_pred_value` (residual on
  top of Boltz scalar) to see whether embeddings add information beyond what
  Boltz already encodes scalarly.
- Investigate why CASR/DRD3 embeddings beat raw Boltz so decisively but
  ADRA2B/MTR1A fail — may correlate with refolding accuracy discussed in the
  paper.
