# Boltz-2 Affinity Embedding Modeling

This project builds per-target models that predict ULVSH ligand affinity from
Boltz-2 affinity-module embeddings, and benchmarks them against Boltz-2's own
scalar outputs. Part 1 data is organized by processing stage under
`data/ulvsh/`: imported ULVSH assets are in `source/`, transferred paper-run
inputs are in `reference_boltz/`, and compact model-ready labels, scalar
predictions, and embeddings are in `modeling/`.

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
- LRIP / interaction-profile scoring (feature set, run — see the LRIP results
  section): Ji et al., *Briefings in Bioinformatics* 22(5) 2021
  (`papers/bbab054.pdf`); Niu et al., LRIP-SF (`papers/aef2177_CombinedPDF_v1.pdf`).

## Setup

```powershell
pip install -r requirements.txt
pip install -e .
```

## Run the Current ROCK1 Pipeline

The default feature set is Boltz affinity embeddings. This uses ULVSH labels
for ROCK1 and the files under
`data/ulvsh/modeling/features/boltz_embeddings/ROCK1/`:

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
- `boltz`: six scalar Boltz affinity fields from the compact reference table.
- `ulvsh_scores`: original ULVSH docking and physics score columns only
  (28 numeric features per ligand: glide, vina, MMGB/SA, etc.).
- `combined`: concatenation of the three feature blocks above. For ROCK1 this
  gives 1024 (embeddings) + 6 (Boltz scalars) + ~25 (ULVSH scores) ≈ 1055 cols.
Regardless of feature set, whenever Boltz scalar records are present they are
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
  probability) from `data/ulvsh/modeling/features/boltz_scalars.tsv`. Reports ROC AUC against
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
`active_bool`) — which reaches 0.84 on ROCK1. See `AGENTS.md` for the full
per-target breakdown and caveats.

## LRIP Interaction-Profile Feature Set (results, 2026-07-15)

Per-residue ligand–receptor interaction energies (LRIP / IP-SF, Junmei Wang
lab) for all 10 targets live in `data/ulvsh/modeling/features/lrip/<TARGET>.dat`
(see that dir's `README.md` for format and join notes). Modeled with the same
RF / StratifiedGroupKFold / `cv_roc_auc` methodology as the embeddings, and
compared to the embeddings and raw Boltz on the **same rows**.

```powershell
# Standalone LRIP vs embeddings vs raw Boltz, per target (runs/lrip/)
python scripts/model_lrip.py
# Does LRIP add on top of Boltz? paired increments (runs/lrip_combined/)
python scripts/model_lrip_combined.py
```

**Standalone LRIP is a negative:** median classification AUC **0.612**, below
the embeddings (0.744) and raw Boltz (best-of-B2A/B2C 0.793). It beats raw
Boltz on 1/10 targets and clears the paper's >0.65 bar on 4/10. ROCK1 is again
the standout (0.869) — a favorable target, not representative.

**But LRIP carries real signal — it is just redundant with the embeddings.**
Added to raw Boltz scalars it helps: median AUC 0.666 → **0.699** (+0.057,
helps 7/10; DRD3 +0.21, MTR1A +0.18). Added on top of the learned stack
(embeddings + Boltz scalars) it does essentially nothing: 0.748 → **0.750**
(+0.008, within noise). So Boltz-2's embeddings already subsume the per-residue
interaction information LRIP computes explicitly. Full breakdown and future
work in `AGENTS.md`.

## Peptide Systems (Part 2 — current direction)

The active set is a curated SKEMPI subset of 13 protein–protein systems with
single and combinatorial mutation ΔΔG measurements. Source bundles are grouped
under `data/peptide_systems/source/<PDB>/`; this replaces the BH3/p53 arm as
the working Part-2 data.

Generate mutation-resolved Boltz cofolding inputs with:

```powershell
python scripts/make_boltz_inputs_peptide_systems.py
```

Outputs live under `data/peptide_systems/boltz/inputs/`. The generator produces 1,705
YAMLs (1,692 unique mutants plus 13 WT) from 2,123 mutant measurement rows. It
deduplicates structures while retaining every experimental observation in a
one-to-many `measurements.tsv`; `variants.tsv` provides one row per generated
complex and replicate-aware ΔΔG summaries.

Mutation numbers refer to `.mapping` field 4 (`SEQIDX`), not PDB residue number
field 3. Most systems map directly to FASTA positions; 3S9D requires sequence
alignment because its FASTA contains extra residues. The generator validates
all coordinates and source amino acids.

Every protein entry uses `msa: empty`, so this is intentionally a
single-sequence experiment. The default YAMLs omit `properties.affinity`, and
the local fork's direct affinity parser/masks still only support a
small-molecule ligand binder. Production embeddings for this set therefore
came from the resolved post-hoc path: the structure run retained trunk `z`, and
`scripts/_build_aff_emb.py` pooled the intended binder-group/partner interface
into the normalized modeling dataset at `data/peptide_systems/modeling/`.

That dataset contains the 1,705-row `features/boltz_embeddings.npz`, a
row-order `index.tsv`, the primary one-row-per-structure `labels.tsv`, the
one-row-per-observation replicate `measurements.tsv`, and the system-level
`manifest.tsv`. It is sufficient for embedding-versus-ΔΔG modeling now and for
later combined models once LRIP feature tables exist, joined by
`(system, input_id)`. It does not include the poses/MM-GBSA outputs needed to
generate LRIP itself. Applying the small-molecule-trained representation out of
domain is intentional—the point is to test whether it still carries useful
protein-interface mutation signal.

See `AGENTS.md` for the scientific invariants, generation audit conclusions,
known issues, and planned per-system Ridge/PLS analysis. The exact modeling
file contract is documented in `data/peptide_systems/modeling/README.md`.

Run the initial per-system models with:

```powershell
python -m boltz2_aff.peptide_pipeline --out-dir runs/peptide_systems/ridge
```

The pipeline validates every embedding/label/measurement join, removes the WT
row from mutant evaluation after using it to construct WT-difference
embeddings, and fits one nested-CV Ridge model per system and feature view.
Default views compare mutation identity, `pair_mean`, `head_mean`, and each
embedding block combined with the mutation baseline. Inner folds select the
Ridge penalty by MAE; saved out-of-fold `metrics.tsv` and `predictions.tsv`
report Spearman, delta-delta-G sign agreement, MAE, RMSE, Pearson, and R2. Use
`--label mean` for the planned median-versus-mean sensitivity pass.

For a per-system linear model with an intercept, subtracting one shared WT
embedding is an affine translation of the raw embeddings, not an independent
source of signal. The explicit WT-difference representation is retained for
interpretability and compatibility with future nonlinear, cross-system, and
LRIP-combined analyses.

## Suggested Peptide Systems (BH3 / p53 — retained, on hold)

> Superseded by `data/peptide_systems/` above; kept because these may return.

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
wild-type, not pooled AUC. See `AGENTS.md` "Planned Part 2".

## Part 2 First Results — BH3 / p53 arm (2026-05-22, on hold)

> Earlier peptide-ligand results, retained for reference. Current Part-2 work
> uses `data/peptide_systems/` (see above).

Embeddings for all 2139 cofolded peptide complexes are stored under
`data/peptides/modeling/features/boltz_embeddings/<system>__<receptor>/affinity_<peptide_id>.npz`
(BH3 689×3
receptors + p53 36×2 receptors). Regenerate the label manifests/index, then run:

```powershell
python scripts/make_boltz_inputs_bh3.py        # (re)writes peptide_index + manifests
python scripts/make_boltz_inputs_p53.py
python scripts/analyze_peptide_embeddings.py    # label-free sensitivity probe
python scripts/part2_analysis.py                # within-series Spearman / ΔΔG-magnitude
python scripts/part2_extras.py                  # key sweep, noise ceiling, selectivity
python scripts/part2_raw_boltz_baseline.py      # raw Boltz-2 scalar baseline (needs cofolds)
```

The raw-Boltz scalar baseline (B2-A / B2-C) is the direct Rognan comparison —
does Boltz-2's *own* scalar output track the mutations? Its analysis is built
and verified; it computes once Boltz-2 has been run over the 2139 peptide input
YAMLs (`data/peptides/boltz/inputs/*/*/input/`) so the affinity JSONs exist. Until
then the script reports what is missing.

Outputs land in `runs/peptide_embeddings/`. First findings — **embedding-model
arm only** (no raw-Boltz scalar baseline yet; peptide affinity JSONs not
extracted):

- **BH3** (n=689/receptor): a cross-validated regressor on the embeddings ranks
  the mutational series with Spearman **0.66** (Bcl-xL), **0.77** (Mcl-1),
  **0.79** (Bfl-1), all p ≤ 1e-86. Holds under within-background CV (Bim
  0.69/0.78/0.77, PUMA 0.27/0.66/0.75). Weak spot: PUMA-background variants on
  Bcl-xL (0.27).
- **p53**: too few point mutants for a supervised model, so the model-free
  probe is the headline — embedding shift-from-WT tracks measured |ΔΔG| with
  Spearman **0.80–0.92** (PMI) and **0.65–0.72** (p53 17–28). A WT-anchored
  ΔΔG-sign check gets the direction right on **every clear effect**
  (|ΔΔG| ≥ 1 kcal/mol; sign agreement 1.00 in 6/8 series).

The mutational signal is present in the representation feeding Boltz-2's scalar
affinity heads — contrasting with the Rognan finding that the raw scalars were
mutation-insensitive. See `AGENTS.md` "Part 2 first results" for the full
breakdown, caveats, and open items.

Follow-ups (`part2_extras.py`): against the SORTCERY **replicate noise ceiling**
(0.83/0.96/0.92 for Bcl-xL/Mcl-1/Bfl-1) the model recovers ~79–86% of the
achievable ranking signal; and the embeddings capture Bcl-2-family
**selectivity** (predicted vs measured receptor preference, Spearman 0.67–0.77,
n=689). See `AGENTS.md` "Part 2 extras".

## Notes for Future Work

See `AGENTS.md` for AI-facing project context, the LRIP feature-set plan,
the Part 2 peptide/mutation plan, current caveats, and an architecture
overview.

Add a top-level `notebooks/` directory later for exploratory visualization,
model diagnostics, and publication figures. Model fitting and metric
generation should remain in reproducible package/script code; notebooks should
read saved out-of-fold predictions and summary tables from `runs/`.
