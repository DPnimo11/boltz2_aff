# Boltz-2 Affinity Embedding Modeling

Per-target models that predict ligand/peptide binding from **Boltz-2
affinity-module embeddings**, benchmarked against Boltz-2's own scalar outputs
and (Part 1) against a physically-motivated LRIP feature set. Two parts:

- **Part 1 — ULVSH small molecules (closed).** Per-target classification/
  regression on 10 ULVSH targets from Boltz-2 embeddings, vs raw Boltz-2 and vs
  LRIP interaction profiles. Honest result: the learned embeddings roughly match
  (and do not decisively beat) raw Boltz-2 in aggregate; LRIP adds real signal
  over raw scalars but is largely redundant with the embeddings.
- **Part 2 — peptide/mutation robustness (active).** Does Boltz-2 track the
  *effect of mutations* on binding? Current data is a 13-system SKEMPI subset
  under `data/peptide_systems/`. An earlier BH3/p53 peptide-ligand arm is on
  hold under `data/peptides/`.

The embedding exporter lives in the neighboring Boltz fork (`../boltz/README.md`):
it writes `affinity_embeddings_<ligand>.npz` files holding the pooled affinity
representation from immediately before the Boltz-2 scalar affinity heads.

See `AGENTS.md` for the deep project context, results, and caveats.

## Data Sources

- ULVSH affinity labels, docking scores, structures:
  https://lab.drugdesign.unistra.fr/datasets/ulvsh/
- Boltz-2 paper input files (published examples/benchmarks, not the ULVSH label
  source): https://zenodo.org/records/16946890
- Methodology reference: Bret, Sindt, Rognan, *Assessing Boltz-2 Performance for
  the Binding Classification of Docking Hits*, J. Chem. Inf. Model. 2026, 66,
  1511-1521. PDF in `papers/`. Per-target ROC AUC of raw Boltz-2 vs ULVSH
  active/inactive labels; AUC > 0.65 "acceptable", median across 10 targets 0.763.
- Boltz-2 model: Passaro et al., *Boltz-2*, bioRxiv 2025.06.14.659707. PDF in
  `papers/`.
- LRIP / interaction-profile scoring: Ji et al., *Briefings in Bioinformatics*
  22(5) 2021 (`papers/bbab054.pdf`); Niu et al., LRIP-SF
  (`papers/aef2177_CombinedPDF_v1.pdf`).

## Setup

```powershell
pip install -r requirements.txt
pip install -e .
```

## Directory Layout

```
boltz2_aff/                 repository root
├── boltz2_aff/             importable Python package
│   ├── data.py             ULVSH label loader, p_affinity derivation
│   ├── features.py         embedding/scalar discovery, feature-set selection
│   ├── modeling.py         train_classifier / train_regressor / boltz_baseline_metrics
│   ├── pipeline.py         Part-1 CLI driver  (python -m boltz2_aff.pipeline)
│   └── peptide_pipeline.py Part-2 per-system nested-CV Ridge CLI
├── scripts/
│   ├── sweep_embedding_keys.py   Part-1 embedding-component sweep
│   ├── nested_cv.py              Part-1 nested-CV (unbiased combo selection)
│   ├── model_lrip.py             Part-1 LRIP vs embeddings vs raw Boltz
│   ├── model_lrip_combined.py    Part-1 "does LRIP add on top of Boltz?" increments
│   ├── make_boltz_inputs_peptide_systems.py   active Part-2 input generator
│   ├── _build_aff_emb.py         post-hoc embedding reconstruction from trunk z
│   └── parse_*/make_*/part2_*.py on-hold BH3/p53 Part-2 ingest + analysis
├── data/
│   ├── ulvsh/                              Part 1
│   │   ├── source/<TARGET>/               imported ULVSH: raw/{vitro,scores}.tsv, minimized/<ZINC>/
│   │   ├── reference_boltz/inputs/        transferred paper-run YAMLs + job file
│   │   └── modeling/                      model-ready inputs
│   │       ├── labels.tsv                 one normalized label row per (target, ligand_id)
│   │       ├── manifest.tsv               per-target coverage
│   │       └── features/
│   │           ├── boltz_scalars.tsv                B2-A/B2-C scalar table (all variants)
│   │           ├── boltz_embeddings/<TARGET>/       affinity_embeddings_<ligand>.npz
│   │           └── lrip/<TARGET>.dat                per-residue interaction profiles (+ README)
│   ├── peptide_systems/                   active Part 2 (13-system SKEMPI subset)
│   │   ├── source/<PDB>/                  curated structures, FASTA, mutation tables
│   │   ├── boltz/inputs/<system>/         generated cofold YAMLs + measurement manifests
│   │   └── modeling/                      consolidated embeddings + labels (see its README)
│   └── peptides/                          on-hold BH3/p53 arm (source, boltz inputs, embeddings)
├── runs/                    all model outputs (git-tracked summaries)
│   ├── sweep_embeddings/  sweep_combined/     Part-1 embedding sweeps
│   ├── lrip/  lrip_combined/                  Part-1 LRIP results
│   ├── peptide_systems/ridge/                 active Part-2 results
│   └── peptide_embeddings/                    on-hold Part-2 results
├── papers/                  reference PDFs (incl. papers/peptides/ for Part 2)
├── AGENTS.md                deep AI-facing project context + full results
└── README.md               this file
```

Part 1 data is organized by processing stage under `data/ulvsh/`: imported
assets in `source/`, transferred paper-run inputs in `reference_boltz/`, and
compact model-ready labels/scalars/embeddings/LRIP in `modeling/`.

## Run the Part-1 Pipeline

Default feature set is Boltz affinity embeddings. Run one, several, or all 10
targets with `--targets` (omit for all):

```powershell
python -m boltz2_aff.pipeline --targets ROCK1 --out-dir runs/rock1_embeddings
python -m boltz2_aff.pipeline --out-dir runs/all_embeddings          # all targets
```

Outputs under `--out-dir`:

- `dataset.csv` — merged labels, metadata, selected feature columns.
- `manifest.json` — data coverage, target/variant counts, metrics summary
  (including the raw Boltz-2 baseline AUC on the same rows).
- `models/classifier.joblib`, `models/regressor.joblib`.
- `models/metrics_*.json` — cross-validation metrics.
- `models/predictions_*.csv` — cross-validated predictions when possible.

## Feature Sets

Use `--feature-set` to choose the modeling input:

- `embeddings` (default): flattened `affinity_embeddings_*.npz` arrays only.
- `boltz`: six scalar Boltz affinity fields from the compact reference table.
- `ulvsh_scores`: original ULVSH docking/physics score columns (~25-28 numeric
  features per ligand: glide, vina, MMGB/SA, etc.).
- `combined`: concatenation of the three blocks above (~1055 cols on ROCK1).

Whenever Boltz scalar records are present they are also merged into `dataset.csv`
as metadata, so the raw Boltz-2 baseline ROC AUC is reported on the same rows
regardless of feature set. (LRIP is evaluated separately — see below.)

```powershell
python -m boltz2_aff.pipeline --feature-set boltz        --out-dir runs/boltz_scalar
python -m boltz2_aff.pipeline --feature-set ulvsh_scores --out-dir runs/ulvsh_scores
python -m boltz2_aff.pipeline --feature-set combined     --out-dir runs/combined
```

## Selecting Embedding Components

Each `affinity_embeddings_<ligand>.npz` contains four arrays from the Boltz-2
affinity module:

| Key | Dims | Description |
|---|---|---|
| `pair_mean1` | 128 | Pooled receptor-ligand interface pair representation, ensemble member 1, before the scalar heads. |
| `pair_mean2` | 128 | Same pooling, ensemble member 2. |
| `head1` | 384 | Representation after the final affinity MLP, ensemble member 1, before the scalar prediction heads. |
| `head2` | 384 | Same, ensemble member 2. |

By default all four are concatenated (1024 dims). Restrict with `--embedding-keys`:

```powershell
python -m boltz2_aff.pipeline --embedding-keys pair_mean1 --out-dir runs/pm1
```

There is **no universal best component** across targets; `pair_mean1` is a
reasonable fixed default. See the sweep section below.

## Modeling Tasks and Metrics

The pipeline fits two tasks by default:

- **Classification** uses the ULVSH `Active` column. Rows with nonnumeric or
  percent-style affinity (`<40%`) are kept here.
- **Regression** uses only uncensored numeric affinity (`Ki`/`EC50`/`IC50`/`Kd`
  or provided `pki`) and trains on `p_affinity = 6 - log10(value_uM)` so larger
  = stronger binding.

Rows are grouped by `target::ligand_id` in cross-validation so multiple Boltz
variants of one ligand cannot leak across folds. Classification uses a no-PCA
`RandomForestClassifier`; regression uses `RidgeCV` in Boltz-residual mode. Key
reported metrics:

- **`classification.cv_roc_auc`** — the headline metric (the paper's primary).
- **`regression.cv_roc_auc`** — a *screening* AUC: per fold train the regressor
  on uncensored-affinity rows, predict `p_affinity` for every active-labeled test
  row (incl. censored inactives), AUC vs `active_bool`. Mirrors the paper.
- **`boltz_baseline`** — raw Boltz-2 per-target AUC (no model fit): B2-A
  `boltz_affinity_pred_value` (lower = stronger) and B2-C
  `boltz_affinity_probability_binary` (higher = active). This is the bar to beat.

## Part 1 Results

### Per-target embedding sweep

`scripts/sweep_embedding_keys.py` runs the pipeline per target per embedding
combination; `scripts/nested_cv.py` adds unbiased inner-fold combo selection.

```powershell
python scripts/sweep_embedding_keys.py --feature-set combined --out-root runs/sweep_combined
python scripts/nested_cv.py --out runs/nested_cv.json
```

**Honest headline:** the learned embeddings roughly match raw Boltz-2 and do not
decisively beat it in aggregate. With nested CV, the combined feature set beats
raw B2-C on 8/10 targets (median AUC ~0.80 vs 0.74), but against the *better* of
B2-A/B2-C per target the margin largely disappears. ROCK1 is the standout
(~0.90) but is a favorable, unrepresentative target. Persistent failures at tiny
n: ADRA2B (n=13, noise) and MTR1A (n=36). Regression is only meaningful for
ROCK1 and CASR; four targets have no uncensored numeric rows and are skipped.
Full per-target tables and caveats in `AGENTS.md`.

### LRIP interaction-profile feature set (2026-07-15)

Per-residue ligand–receptor interaction energies (LRIP / IP-SF, Junmei Wang lab)
for all 10 targets in `data/ulvsh/modeling/features/lrip/<TARGET>.dat` (format +
join notes in that dir's README). Scored with the same RF / StratifiedGroupKFold
/ `cv_roc_auc` methodology as the embeddings, compared on the **same rows**.

```powershell
python scripts/model_lrip.py            # standalone LRIP vs emb vs raw Boltz  -> runs/lrip/
python scripts/model_lrip_combined.py   # does LRIP add on top of Boltz?       -> runs/lrip_combined/
```

**Standalone LRIP is a negative:** median AUC **0.612**, below the embeddings
(0.744) and raw Boltz (best-of-B2A/B2C 0.793); clears >0.65 on 4/10. ROCK1 again
the standout (0.869).

**But LRIP carries real signal — it is just redundant with the embeddings.**
Added to raw Boltz scalars it helps (median 0.666 → **0.699**, +0.057, helps
7/10). Added on top of the learned stack (embeddings + Boltz scalars) it does
essentially nothing (0.748 → **0.750**, +0.008, within noise). Boltz-2's
embeddings already subsume the per-residue interaction signal LRIP computes
explicitly. Full breakdown and future work in `AGENTS.md`.

## Part 2 — Peptide / Mutation Robustness (active)

The active set is a curated SKEMPI subset of **13 protein–protein systems** with
single and combinatorial mutation ΔΔG measurements, under
`data/peptide_systems/source/<PDB>/`. It replaces the BH3/p53 arm.

```powershell
python scripts/make_boltz_inputs_peptide_systems.py   # -> data/peptide_systems/boltz/inputs/
python -m boltz2_aff.peptide_pipeline --out-dir runs/peptide_systems/ridge
```

The generator produces 1,705 sequence-only cofold YAMLs (1,692 unique mutants +
13 WT) from 2,123 mutant measurement rows, deduplicating structures while keeping
every observation in a one-to-many `measurements.tsv`. Production embeddings came
from the post-hoc path (`scripts/_build_aff_emb.py` pools the binder/partner
interface from retained trunk `z`) into `data/peptide_systems/modeling/`; that
bundle does **not** include poses/MM-GBSA, so no LRIP or raw-scalar baseline for
peptides yet. The per-system pipeline validates every embedding/label/measurement
join, uses WT only to build WT-difference embeddings, and fits one nested-CV
Ridge model per system and feature view; out-of-fold `metrics.tsv`/`predictions.tsv`
report Spearman, ΔΔG-sign agreement, MAE, RMSE, Pearson, R². Evaluation is
within-series Spearman and ΔΔG sign, not pooled AUC. Contract in
`data/peptide_systems/modeling/README.md`; scientific detail in `AGENTS.md`.

### On-hold BH3/p53 arm

An earlier peptide-ligand arm (BH3 ↔ Mcl-1/Bcl-xL/Bfl-1, p53 TAD ↔ MDM2/MDMX)
lives under `data/peptides/` with 2,139 extracted embeddings. First results
(embedding-model arm only): within-series Spearman 0.66–0.79 on BH3, |ΔΔG|
magnitude Spearman 0.65–0.92 on p53, and captured Bcl-2-family selectivity — the
mutational signal *is* present in the representation feeding Boltz-2's scalar
heads. The raw-Boltz scalar baseline (the direct Rognan comparison) is built but
blocked on running Boltz-2 over the peptide YAMLs. Retained but not active; full
detail in `AGENTS.md`.

## Notes for Future Work

See `AGENTS.md` for the full per-target breakdowns, caveats, LRIP future work,
and the Part-2 plan. A top-level `notebooks/` directory can be added later for
exploratory visualization and figures — model fitting and metrics should stay in
reproducible package/script code, with notebooks reading saved out-of-fold
predictions and summaries from `runs/`.
