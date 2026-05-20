# Boltz-2 Affinity Embedding Modeling

This project builds per-target models that predict ULVSH ligand affinity from
Boltz-2 affinity-module embeddings, and benchmarks them against Boltz-2's own
scalar outputs. Labels and docking score features come from `data/ULVSH`;
Boltz scalar affinity JSONs and embedding exports are read from `data/Boltz-2`;
the currently extracted ROCK1 embeddings are in `targets/ROCK1`.

The embedding extraction behavior is documented in the neighboring Boltz fork at
`../boltz/README.md`. In short, that fork writes `affinity_embeddings_<ligand>.npz`
files containing pooled affinity representations from immediately before the
Boltz-2 scalar affinity heads.

## Data Sources

- ULVSH affinity labels, docking scores, and structure files:
  https://lab.drugdesign.unistra.fr/datasets/ulvsh/
- Boltz-2 paper input files used to run the published examples/benchmarks
  (not the ULVSH affinity-label source):
  https://zenodo.org/records/16946890
- Methodology reference: Bret, Sindt, Rognan, *Assessing Boltz-2 Performance
  for the Binding Classification of Docking Hits*, J. Chem. Inf. Model. 2026,
  66, 1511-1521. PDF in `papers/`.
- Boltz-2 model: Passaro et al., *Boltz-2: Towards Accurate and Efficient
  Binding Affinity Prediction*, bioRxiv 2025.06.14.659707. PDF in `papers/`.
- LRIP / interaction-profile scoring (planned feature set): Ji et al.,
  *Briefings in Bioinformatics* 22(5) 2021 (`papers/bbab054.pdf`); Niu et al.,
  LRIP-SF (`papers/aef2177_CombinedPDF_v1.pdf`).

## Setup

```powershell
pip install -r requirements.txt
pip install -e .
```

## Run the Current ROCK1 Pipeline

The default feature set is Boltz affinity embeddings. This uses ULVSH labels
for ROCK1 and the existing `targets/ROCK1` embedding files:

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 --out-dir runs/rock1_embeddings
```

Outputs under `runs/rock1_embeddings`:

- `dataset.csv`: merged labels, metadata, and selected feature columns.
- `manifest.json`: data coverage, target/variant counts, and metrics summary
  (including the raw Boltz-2 baseline AUC for the same rows — see below).
- `models/classifier.joblib`, `models/regressor.joblib`.
- `models/metrics_*.json`: cross-validation metrics.
- `models/predictions_*.csv`: cross-validated predictions when possible.

## Feature Sets

Use `--feature-set` to choose the modeling input:

- `embeddings` (default): flattened `affinity_embeddings_*.npz` arrays only.
- `boltz`: scalar Boltz affinity JSON fields only (6 numeric scalars per ligand).
- `ulvsh_scores`: original ULVSH docking and physics score columns only
  (28 numeric features per ligand: glide, vina, MMGB/SA, etc.).
- `combined`: concatenation of the three feature blocks above. For ROCK1 this
  gives 1024 (embeddings) + 6 (Boltz scalars) + ~25 (ULVSH scores) ≈ 1055 cols.
Regardless of feature set, whenever Boltz scalar JSONs are present they are
also merged into `dataset.csv` as metadata columns so the pipeline can report
the raw Boltz-2 baseline ROC AUC against the same rows.

Examples:

```powershell
python -m boltz2_aff.pipeline --feature-set boltz --out-dir runs/boltz_scalar
python -m boltz2_aff.pipeline --feature-set ulvsh_scores --out-dir runs/ulvsh_scores
python -m boltz2_aff.pipeline --feature-set combined --out-dir runs/combined
```

## Selecting Embedding Components

Each `affinity_embeddings_<ligand>.npz` file contains four arrays from the
Boltz-2 affinity module:

| Key | Dimensions | Description |
|---|---|---|
| `pair_mean1` | 128 | Pooled receptor-ligand interface pair representation, ensemble member 1, before the scalar heads. |
| `pair_mean2` | 128 | Same pooling, ensemble member 2. |
| `head1` | 384 | Representation after the final affinity MLP, ensemble member 1, before the scalar prediction heads. |
| `head2` | 384 | Same, ensemble member 2. |

By default all four are concatenated (1024 dims). Restrict to a subset with
`--embedding-keys`:

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 --embedding-keys pair_mean1 --out-dir runs/rock1_pm1
python -m boltz2_aff.pipeline --targets ROCK1 --embedding-keys pair_mean1 pair_mean2 --out-dir runs/rock1_pmboth
```

On ROCK1 (n=68 active labels, n=27 numeric Ki/IC50/Kd), restricted subsets
*beat* the full 1024-dim concatenation — see `runs/rock1_sweep/` and the
sweep section below.

## Modeling Tasks

The pipeline fits two tasks by default:

- **Classification** uses the ULVSH `Active` column. Rows with nonnumeric or
  percent-style affinity values such as `<40%` are kept here.
- **Regression** uses only uncensored numeric affinity measurements (`Ki`,
  `EC50`, `IC50`, `Kd`, or provided `pki`) and trains on
  `p_affinity = 6 - log10(value_uM)` so larger values mean stronger binding.

Rows are grouped by `target::ligand_id` during cross-validation so multiple
Boltz variants for the same ligand do not leak across folds.

### Metrics

For each task the manifest reports cross-validated metrics. New since recent
changes:

- **`regression.cv_roc_auc`**: a *screening* AUC. Per fold the regressor trains
  on uncensored-affinity rows in the train split, then predicts `p_affinity`
  for every active-labeled test row (including inactives that have no numeric
  affinity). AUC is computed over the concatenated predictions against
  `active_bool`. This answers "can the regressor rank actives above
  inactives?", matching what the Bret/Sindt/Rognan paper reports.
- **`boltz_baseline`**: raw Boltz-2 scalar baselines per target (no model
  fit), computed by reading `boltz_affinity_pred_value` (B2-A, lower =
  stronger binder) and `boltz_affinity_probability_binary` (B2-C, 0–1 binding
  probability) directly from the affinity JSONs. Reports ROC AUC against
  `active_bool` and Pearson/Spearman against `p_affinity`. This is the bar
  the embedding models need to beat.

## Per-Target Embedding Sweep (current findings)

`scripts/sweep_embedding_keys.py` runs the pipeline once per target per
embedding combination and writes `summary.json`, `summary_by_target.json`,
and `medians.json` (paper-style median across targets).

```powershell
python scripts/sweep_embedding_keys.py --feature-set embeddings --out-root runs/sweep_embeddings
python scripts/sweep_embedding_keys.py --feature-set combined  --out-root runs/sweep_combined
```

**Headline result (no-PCA RF, nested CV, all 10 targets):**

The combined feature set (embeddings + Boltz scalars + ULVSH scores) with
inner-fold combo selection beats raw Boltz-2 B2-C on **8/10 targets**, median
classification AUC **0.798 vs 0.740**. With a single fixed `pair_mean1` combo,
7/10 targets win (median 0.803 vs 0.740). Removing the adaptive PCA from the
RF classifier was the decisive change — it recovered ROCK1 from 0.808 to 0.912
and restored results across most targets.

```powershell
# Reproduce the sweep (all 10 targets, 9 embedding combos each)
python scripts/sweep_embedding_keys.py --feature-set combined --out-root runs/sweep_combined_v2

# Reproduce the nested CV (outer/inner combo selection, unbiased estimate)
python scripts/nested_cv.py --out runs/nested_cv.json
```

Persistent failures: ADRA2B (n=13, noise-dominated) and MTR1A (n=36, likely
poor cofold pose quality). There is **no universal best embedding component** —
it varies by target; `pair_mean1` is a reasonable default.

**Regression** is a secondary result. Four targets (CNR1, DRD3, SC6A4, SGMR2)
have no uncensored numeric affinity rows and are skipped. Of the remainder,
only ROCK1 (n=27, Pearson ~0.70) and CASR (n=148, Pearson ~0.50) give
meaningful signal. The more informative metric is `regression.cv_roc_auc` —
the screening AUC (train on actives-only rows, rank all test rows against
`active_bool`) — which reaches 0.84 on ROCK1. See `context.md` for the full
per-target breakdown and caveats.

## Suggested Peptide Systems (planned Part 2)

For the peptide/mutation robustness study (does Boltz-2 track mutational
effects?), candidate systems with ≥30 quantified mutational variants of one
peptide against one receptor:

| System | Peptide | Receptor(s) | Data source | Notes |
|---|---|---|---|---|
| **BH3 (top pick)** | ~26-mer α-helix (BIM/BID/PUMA bg) | MCL-1, BCL-xL | Keating lab — Jenson et al., *eLife* 2018 and related affinity datasets | Hundreds of single + combinatorial variants, SPR/FP Kd, wide dynamic range, dual receptor (selectivity) |
| **p53 TAD** | p53(17–28) helix + pDI/PMI variants | MDM2, MDMX | Curated literature Kd; phage-optimized variants | Small well-defined interface; dozens of variants |
| **MHC-I nonamer** | 9-mer (e.g. NLVPMVATV, GILGFVFTL) | HLA-A*02:01 | IEDB / NetMHCpan | Thousands of single substitutions; mixed assays — scale stress-test |
| **PDZ / CRIPT** | C-terminal peptide | PSD-95 PDZ3 | Peptide saturation-mutagenesis literature | Short interface, low dynamic range — harder secondary case |

Evaluation for Part 2 is within-series Spearman and ΔΔG sign agreement vs
wild-type, not pooled AUC. See `context.md` "Planned Part 2".

## Notes for Future Work

See `context.md` for AI-facing project context, the LRIP feature-set plan,
the Part 2 peptide/mutation plan, current caveats, and an architecture
overview.
