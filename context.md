# Project Context

This repository models ULVSH ligand affinity using internal Boltz-2 affinity-module
embeddings exported by the neighboring `../boltz` fork.

## Current Data Layout

- `data/ULVSH/<target>/raw/vitro.tsv` contains labels: ligand ID,
  target-specific affinity/activity measurement, and active/inactive status.
- `data/ULVSH/<target>/raw/scores.tsv` contains the original ULVSH docking and
  physics score features.
- `data/Boltz-2/<target>/<variant>/output/<ligand>/affinity_<ligand>.json`
  contains scalar Boltz affinity predictions.
- Future Boltz embedding exports should appear as
  `affinity_embeddings_<ligand>.npz` next to the scalar affinity JSON.
- The currently available extracted embedding set is
  `targets/ROCK1/affinity_embeddings_*.npz`.

## Embedding Provenance

The `../boltz` fork exports affinity embeddings from the Boltz-2 affinity module.
The exported arrays are documented in `../boltz/README.md`:

- `affinity_embedding_pair_mean`: pooled receptor-ligand/interface pair
  representation before the final affinity MLP.
- `affinity_embedding_head`: representation after the final affinity MLP and
  before scalar prediction heads.
- Ensemble runs append `1` and `2` suffixes to both arrays.

This repository flattens all exported arrays into numeric columns prefixed with
`emb_`.

## Modeling Defaults

- Classification uses the ULVSH `Active` column and can include rows whose
  affinity/activity measurement is nonnumeric or percent-style, such as `<40%`.
- Regression uses only uncensored numeric affinity measurements (`Ki`, `EC50`,
  `IC50`, `Kd`, or provided `pki`) and trains on
  `p_affinity = 6 - log10(value_uM)`, so larger values mean stronger binding.
- Cross-validation groups rows by `target::ligand_id` to avoid leakage across
  Boltz variants for the same ligand.
- Default feature set is `embeddings`. Other options are `boltz`,
  `ulvsh_scores`, and `combined`.

## Main Command

Run the current ROCK1 embedding pipeline:

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 --out-dir runs/rock1_embeddings
```

Run all targets once embedding exports exist under `data/Boltz-2`:

```powershell
python -m boltz2_aff.pipeline --feature-set embeddings --out-dir runs/all_embeddings
```

Compare with scalar Boltz predictions or original ULVSH scores:

```powershell
python -m boltz2_aff.pipeline --feature-set boltz --out-dir runs/boltz_scalar
python -m boltz2_aff.pipeline --feature-set ulvsh_scores --out-dir runs/ulvsh_scores
python -m boltz2_aff.pipeline --feature-set combined --out-dir runs/combined
```

## Outputs

Each run writes:

- `dataset.csv`: model-ready merged rows and selected features.
- `manifest.json`: data coverage, target/variant counts, and metrics summary.
- `models/classifier.joblib` and/or `models/regressor.joblib`.
- `models/metrics_classification.json` and/or `models/metrics_regression.json`.
- Cross-validated prediction CSVs when cross-validation is possible.

## Caveats

- At the time this context was written, `data/Boltz-2` has scalar affinity JSONs
  but no `affinity_embeddings_*.npz` files. Only `targets/ROCK1` has extracted
  embedding files.
- `targets/ROCK1` has 68 embedding files for 69 ULVSH ROCK1 labels; `mol_44`
  is currently missing embeddings.
- The default models are deliberately conservative linear baselines:
  L2-regularized logistic regression for active/inactive classification and
  ridge regression for `p_affinity`.
