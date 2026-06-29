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
`active_bool`) — which reaches 0.84 on ROCK1. See `AGENTS.md` for the full
per-target breakdown and caveats.

## Peptide Systems (Part 2 — current direction)

The active Part-2 system set lives in `peptide_systems/`. These are **classic
protein–protein complexes with experimental point-mutation ΔΔG data** (alanine
and non-alanine single mutants plus combinatorial variants), curated from
the binding-affinity mutagenesis literature. This set **replaces the BH3/p53
peptide-ligand arm** as the working Part-2 data — the BH3/p53 material below is
retained (it may return later) but is on hold.

13 systems are present (PDB id + the two interacting chain groups in the file
name, e.g. `1A22_A_B` = human growth hormone chain A ↔ hGH-receptor chain B):
1A22, 1AO7, 1BRS, 1CHO, 1GC1, 1JTG, 1VFB, 2B2X, 3BT1, 3HFM, 3S9D, 3SE3, 4G0N.

Per-system files (`<PDB>_<grp1>_<grp2>` denotes the measured chain partition):

| File | Contents |
|---|---|
| `<PDB>.pdb` | Complex structure. |
| `<PDB>.fasta` | Chain sequences (regenerated by the `bat` csh helper → `pdb2fasta`). |
| `<PDB>.mapping` | One line per residue: `RESNAME CHAIN PDBNUM SEQIDX`. Resolves mutation positions to PDB numbering. |
| `<PDB>_<g1>_<g2>.txt` | Original raw mutational records. |
| `<PDB>_<g1>_<g2>_New.txt` | **Curated table — use this.** Schema below. |
| `<PDB>_<g1>_<g2>_New.xlsx` | Excel mirror of `_New.txt`. |
| `<PDB>_<g1>_<g2>.wt` | (some systems) WT Kd sub-series as `Kd  count` lines. |

### `_New.txt` column schema (example: `1A22_A_B_New.txt`)

Tab-separated, one header row, one `WT` reference row, then one row per variant,
followed by two non-data footer lines.

| Column | Example | Meaning |
|---|---|---|
| `ID` | `1A22_3` | Row id `<PDB>_<n>`. Row 1 is the `WT` reference. |
| `Mutation` | `HA18A` | Substitution(s) as `<wtAA><chain><resnum><mutAA>` (His18 of chain A → Ala). `WT` on the reference row. Multi-point variants are comma-joined and quoted (`"KA157A,EA163A"`). Not only alanine scans — non-Ala targets occur (`FB55S`, `RB11L`). Positions follow `.mapping`/PDB numbering. |
| `DG` | `-13.2024` | Mutant binding free energy (kcal/mol) re-anchored to **one global WT reference**: `DG = DG_WTref + (DGmut − DGWT)`. `DG_WTref` is the WT-row value (1A22: −12.7160) and is repeated alone in the footer. This puts every variant on a common WT baseline even though sub-series have different absolute WT affinities. |
| `Activity_Mutate` | `3.96E-10` | Measured mutant dissociation constant Kd (M). `0` on the WT row. |
| `DGmut` | `-12.8264` | ΔG from the mutant Kd: `RT·ln(Kd)` ≈ `0.5925·ln(Kd)` at 298 K (kcal/mol). `0` on the WT row. |
| `Activity_WT` | `9.00E-10` | Measured WT Kd (M) for **that mutation's experimental sub-series**. A system can have several WT sub-series (see the `.wt` file); 1A22 mixes 9.0e-10, 3.4e-10, 4.4e-10, … `0` on the WT row. |
| `DGWT` | `-12.3400` | ΔG of that sub-series WT: `RT·ln(Activity_WT)`. `0` on the WT row. |

The per-variant effect is **ΔΔG = `DGmut − DGWT`** (> 0 weakens binding,
< 0 strengthens it). Footer lines: (1) a lone value in the `DG` column =
`DG_WTref`, the global WT anchor; (2) `>1E-06  1.00E-06` = the weak-binding
detection cap — any Kd above 1 µM is reported at the 1 µM limit.

## Suggested Peptide Systems (BH3 / p53 — retained, on hold)

> Superseded by `peptide_systems/` above; kept because these may return.

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
> uses `peptide_systems/` (see above).

Embeddings for all 2139 cofolded peptide complexes are stored under
`targets/peptides/<system>__<receptor>/affinity_<peptide_id>.npz` (BH3 689×3
receptors + p53 36×2 receptors). Regenerate the label manifests/index, then run:

```powershell
python scripts/make_boltz_inputs_bh3.py        # (re)writes peptide_index + manifests
python scripts/make_boltz_inputs_p53.py
python scripts/analyze_peptide_embeddings.py    # label-free sensitivity probe
python scripts/part2_analysis.py                # within-series Spearman / ΔΔG-magnitude
```

Outputs land in `runs/peptide_embeddings/`. First findings — **embedding-model
arm only** (no raw-Boltz scalar baseline yet; peptide affinity JSONs not
extracted):

- **BH3** (n=689/receptor): a cross-validated regressor on the embeddings ranks
  the mutational series with Spearman **0.66** (Bcl-xL), **0.77** (Mcl-1),
  **0.79** (Bfl-1), all p ≤ 1e-86. Weak spot: PUMA-background variants on
  Bcl-xL (0.30).
- **p53**: too few point mutants for a supervised model, so the model-free
  probe is the headline — embedding shift-from-WT tracks measured |ΔΔG| with
  Spearman **0.80–0.92** (PMI) and **0.65–0.72** (p53 17–28).

The mutational signal is present in the representation feeding Boltz-2's scalar
affinity heads — contrasting with the Rognan finding that the raw scalars were
mutation-insensitive. See `AGENTS.md` "Part 2 first results" for the full
breakdown, caveats, and open items.

## Notes for Future Work

See `AGENTS.md` for AI-facing project context, the LRIP feature-set plan,
the Part 2 peptide/mutation plan, current caveats, and an architecture
overview.
