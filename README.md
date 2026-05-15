# Boltz-2 Affinity Embedding Modeling

This project builds models that predict ULVSH ligand affinity from Boltz-2
affinity-module embeddings. Labels and optional baseline score features come
from `data/ULVSH`; Boltz scalar affinity JSONs and future embedding exports are
read from `data/Boltz-2`; the currently extracted ROCK1 embeddings are in
`targets/ROCK1`.

The embedding extraction behavior is documented in the neighboring Boltz fork at
`../boltz/README.md`. In short, that fork writes
`affinity_embeddings_<ligand>.npz` files containing pooled affinity
representations from immediately before the Boltz-2 scalar affinity heads.

## Setup

The code uses standard scientific Python packages:

```powershell
pip install -r requirements.txt
```

You can also install this repository as a small local package:

```powershell
pip install -e .
```

## Run the Current ROCK1 Pipeline

The default feature set is Boltz affinity embeddings. This command uses ULVSH
labels for ROCK1 and the existing `targets/ROCK1` embedding files:

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 --out-dir runs/rock1_embeddings
```

Outputs are written under `runs/rock1_embeddings`:

- `dataset.csv`: merged labels, metadata, and selected feature columns.
- `manifest.json`: data coverage, target/variant counts, and metrics summary.
- `models/classifier.joblib`: active/inactive classifier.
- `models/regressor.joblib`: `p_affinity` regressor.
- `models/metrics_*.json`: cross-validation metrics.
- `models/predictions_*.csv`: cross-validated predictions when possible.

## Feature Sets

Use `--feature-set` to choose the modeling input:

- `embeddings`: flattened `affinity_embeddings_*.npz` arrays.
- `boltz`: scalar Boltz affinity JSON fields.
- `ulvsh_scores`: original ULVSH docking and physics score columns.
- `combined`: embeddings, scalar Boltz predictions, and ULVSH scores.

Examples:

```powershell
python -m boltz2_aff.pipeline --feature-set boltz --out-dir runs/boltz_scalar
python -m boltz2_aff.pipeline --feature-set ulvsh_scores --out-dir runs/ulvsh_scores
python -m boltz2_aff.pipeline --feature-set combined --out-dir runs/combined
```

## Modeling Targets

The pipeline fits two tasks by default:

- Classification uses the ULVSH `Active` column. Rows with nonnumeric or
  percent-style affinity values such as `<40%` are still usable here.
- Regression uses only uncensored numeric affinity measurements (`Ki`, `EC50`,
  `IC50`, `Kd`, or provided `pki`) and trains on
  `p_affinity = 6 - log10(value_uM)`, so larger values mean stronger binding.

Rows are grouped by `target::ligand_id` during cross-validation so multiple
Boltz variants for the same ligand do not leak across folds.

## Full ULVSH Modeling

Once the Boltz extraction fork has written `affinity_embeddings_*.npz` files next
to the `affinity_*.json` outputs in `data/Boltz-2`, run all available targets
with:

```powershell
python -m boltz2_aff.pipeline --feature-set embeddings --out-dir runs/all_embeddings
```

Restrict to specific targets or variants with:

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 CNR2 --variants wt --out-dir runs/wt_subset
```

## Notes for Future Work

See `context.md` for AI-facing project context, current caveats, and expected
data layout.
