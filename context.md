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

## Current ROCK1 Findings (2026-05-15)

68 rows for classification, 27 for regression. Best results vs raw Boltz B2-C
baseline (AUC 0.854, Pearson 0.710, Spearman 0.707):

- `pair_mean1` + `combined`: classifier AUC 0.908. Best classifier.
- `pair_mean1+pair_mean2` + `combined`: regressor Pearson 0.803,
  Spearman 0.809, R² 0.644. Best regressor.
- The full 1024-dim embedding concatenation (`all`) is *worse* than the
  pair-mean subsets on every metric — the 384-dim head components overfit at
  n=27 (`head2` alone hits R² = -11.92).

## Caveats

- Only ROCK1 currently has embedding files. Other 9 ULVSH targets need
  embedding extraction from `../boltz` before they can be modeled. ULVSH-wide
  `manifest.json` reports 875 labels missing embeddings.
- n=27 for regression means CV is noisy: 5 folds = ~5 test rows each. The
  embedding sweep differences should be read as broad trends, not precise
  rankings.
- ULVSH does not provide numeric IC50/Ki for inactives (they're censored or
  percent-style), so the regression subset is all-active. Hence the screening
  AUC implementation that predicts on inactives despite training on actives.

## Possible Next Steps

- Extract embeddings for the other 9 targets so each can be modeled per the
  same pipeline.
- Try PLS regression or PCA→ridge to handle p≫n for the `head` components
  rather than discarding them.
- Train the regressor on `p_affinity − boltz_affinity_pred_value` (residual on
  top of Boltz scalar) to see whether embeddings add information beyond what
  Boltz already encodes scalarly.
- For multi-target sweeps, treat per-target ROC AUC as the primary metric and
  report median across targets as the paper does.
