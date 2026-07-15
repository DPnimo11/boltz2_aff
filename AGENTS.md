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
  - `peptide_pipeline.py` — active Part-2 validated loader and per-system
    nested-CV Ridge modeling CLI.
- `scripts/sweep_embedding_keys.py` — Part-1 embedding-component sweep harness;
  `scripts/nested_cv.py` — unbiased nested-CV combo selection.
- `scripts/model_lrip.py` / `scripts/model_lrip_combined.py` — Part-1 LRIP
  modeling: standalone LRIP vs embeddings vs raw Boltz, and the "does LRIP add on
  top of Boltz?" paired increments. Reuse `modeling.train_classifier` /
  `boltz_baseline_metrics`; run with `PYTHONPATH=.`.
- `scripts/make_boltz_inputs_peptide_systems.py` — active Part-2 generator;
  applies curated mutations to the measured FASTA chain groups, deduplicates
  structures without dropping repeated measurements, and writes Boltz YAMLs
  plus manifests under `data/peptide_systems/boltz/inputs/`.
- `scripts/_build_aff_emb.py` — post-hoc Part-2 embedding reconstruction from
  saved trunk `z`; pools an explicit binder-chain-group/partner interface and
  applies the two checkpoint affinity MLPs.
- `scripts/analyze_peptide_embeddings.py` — label-free Part-2 diagnostic
  (embedding shift under mutation; QC of the extracted set).
- `scripts/part2_analysis.py` — Part-2 within-series Spearman / ΔΔG-magnitude
  analysis (joins peptide embeddings to measured affinity via the manifests).
- `scripts/part2_extras.py` — Part-2 follow-ups needing no new data: embedding-key
  sweep, BH3 replicate noise ceiling, and BH3 cross-target selectivity.
- `scripts/part2_raw_boltz_baseline.py` — Part-2 raw Boltz-2 scalar baseline
  (B2-A / B2-C within-series Spearman + ΔΔG-sign, the Rognan comparison). Built
  and verified; computes once the peptide affinity JSONs are produced.

## Current Data Layout

- `data/ulvsh/source/<target>/raw/vitro.tsv` contains labels: ligand ID,
  target-specific affinity/activity measurement, and active/inactive status.
- `data/ulvsh/source/<target>/raw/scores.tsv` contains the original ULVSH docking and
  physics score features.
- `data/ulvsh/modeling/features/boltz_scalars.tsv` contains all six scalar
  affinity fields from the transferred paper/reference run. The original
  5.7 GB output tree was removed after exact record-level validation. The
  table has 2,830 unique `(target, variant, ligand_id)` rows; consolidation
  corrected 74 CASR shuffled records stored under the transferred typo
  `shuffled/ouput/`, which the legacy path inference had mislabeled as WT.
- Part-1 affinity embeddings produced with the modified exporter live under
  `data/ulvsh/modeling/features/boltz_embeddings/<target>/`. ROCK1 has 68 of
  69 ligands; `mol_44` is missing.
- `data/ulvsh/modeling/features/lrip/<TARGET>.dat` — per-residue LRIP
  interaction-energy matrices (one per target; format + join quirks in that
  dir's `README.md`). See "LRIP Interaction-Profile Feature Set" below.
- `data/ulvsh/modeling/{labels.tsv,manifest.tsv}` provides normalized labels
  and per-target feature/input coverage. The transferred reference YAMLs and
  job file live locally under `data/ulvsh/reference_boltz/`.
- **Active Part 2:** source bundles live under
  `data/peptide_systems/source/<PDB>/`; generated sequence-only Boltz inputs
  and one-to-many measurement manifests live under
  `data/peptide_systems/boltz/inputs/<system>/`.
- **Active Part-2 modeling dataset:**
  `data/peptide_systems/modeling/` contains all 1,705 consolidated
  `pair_mean`/head embeddings, labels, raw observations, row index, manifest,
  and modeling context. It is complete for the primary embedding-versus-ΔΔG
  analysis and was audited on 2026-07-10. `index.tsv` matches NPZ order;
  `labels.tsv` does not, so always join on `(system, input_id)` rather than
  row position.
- **BH3/p53 Part 2 (on hold):** source papers remain under
  `papers/peptides/{bh3,p53,p53_2,HLA_A0201}/`, parsed tables under
  `data/peptides/source/<system>/`, and the 2,139 previously extracted embeddings under
  `data/peptides/modeling/features/boltz_embeddings/`. They use the newer
  `pair_mean`/`head_mean` export schema
  and are deliberately not discovered by the Part-1 pipeline.

## `data/peptide_systems/` — active Part-2 system set (updated 2026-07-10)

This is a curated SKEMPI subset of **13 classic protein–protein complexes with
experimental point-mutation ΔΔG data**: 1A22, 1AO7, 1BRS, 1CHO, 1GC1, 1JTG,
1VFB, 2B2X, 3BT1, 3HFM, 3S9D, 3SE3, 4G0N. It replaces the BH3/p53 peptide arm
as the active Part-2 direction.

### Source layout

Each bundle is isolated under `data/peptide_systems/source/<PDB>/`. The curated
filename stem `<PDB>_<grp1>_<grp2>` identifies the two measured partner chain
groups (for example `2B2X_HL_A` is antibody chains H+L against chain A).

- `<PDB>.pdb` — source complex structure.
- `<PDB>.fasta` — chain sequences regenerated by the tracked `source/bat` csh
  helper via `pdb2fasta`.
- `<PDB>.mapping` — `RESNAME CHAIN PDBNUM SEQIDX`; mutation numbers match
  **field 4 `SEQIDX`**, not field 3 `PDBNUM`.
- `<PDB>_<g1>_<g2>.txt` — original raw mutational records.
- `<PDB>_<g1>_<g2>_New.txt` — **curated source of truth**.
- `<PDB>_<g1>_<g2>_New.xlsx` — Excel mirror of `_New.txt`.
- `<PDB>_<g1>_<g2>.wt` — optional WT-Kd sub-series counts.

`3SE3.fasta` also contains chain C, but the curated measured pair is B–A. The
generator includes only the chain groups named in the filename and records C as
excluded in the manifests.

### Curated table semantics

`_New.txt` is tab-separated: header, one WT row, experimental variant rows, and
two non-data footer lines. Its columns are `ID`, `Mutation`, `DG`,
`Activity_Mutate`, `DGmut`, `Activity_WT`, and `DGWT`.

- `Mutation` uses `<wtAA><chain><SEQIDX><mutAA>`; multi-point variants are
  comma-joined. Non-alanine substitutions and variants of up to 15 mutations
  occur.
- **ΔΔG = `DGmut − DGWT`** (> 0 weakens binding, < 0 strengthens).
- `DG = DG_WTref + ΔΔG` re-anchors every experimental sub-series to the one WT
  value on the system's WT row. The generator verifies this identity within
  table-rounding tolerance.
- Different rows for the same mutant may use different matched WT references.
  Do not average raw Kd. Aggregate on the ΔΔG/free-energy scale if one modeling
  label per structure is needed.
- The footer's `>1E-06  1.00E-06` denotes a weak-binding detection cap. The
  generator preserves numeric labels but does not yet infer censoring.

### Mutation coordinate handling

All 3,711 substitution tokens resolve to their expected wild-type residue via
`SEQIDX`. For 12 systems, the mapping residue sequence equals the FASTA and
`SEQIDX` is the FASTA position. `3S9D` FASTA chains A/B contain respectively
two/five extra residues, so the generator aligns mapping sequence to FASTA and
translates `SEQIDX` before editing. Direct FASTA indexing would silently mutate
the wrong 3S9D residues.

### Input generation

```powershell
python scripts/make_boltz_inputs_peptide_systems.py
```

The generator writes **1,705 YAMLs**: 1,692 unique canonical mutants plus 13
WT complexes, derived from 2,123 mutant measurement rows. There are 431
measurement rows beyond the unique-mutant count, 296 unique mutants with
repeated measurements, and 620 unique multi-mutants.

Per system, `data/peptide_systems/boltz/inputs/<system>/` contains:

- `input/<input_id>.yaml` — one sequence-only cofold per unique mutation set.
- `variants.tsv` — one row per YAML, including replicate count and median/mean/
  spread of ΔΔG.
- `measurements.tsv` — every original curated observation mapped to `input_id`.
- The parent `boltz/inputs/manifest.tsv` summarizes counts and chain decisions.

Every generated protein entry has `msa: empty`; single-sequence execution is an
intentional and resolved part of the experiment.

The YAMLs deliberately omit `properties.affinity` by default. The local
`../boltz` fork **does** contain the custom embedding exporter (`dacb835`): it
returns pre-MLP `g_pair_mean` and post-MLP `g_head`, propagates the ensemble
representations, and writes `affinity_embeddings_<record_id>.npz`. However,
that patch does not extend affinity input parsing or masking to protein binders:

- `schema.py` still requires one string binder of entity type `ligand`.
- `AffinityInfo` / the tokenizer create an affinity mask for one chain.
- The affinity module defines `rec_mask` as all protein tokens, so simply
  marking a protein as binder would make binder and receptor masks overlap.
- The feature path expects an `affinity_mw` value even though MW correction is
  off by default.

Consequently, the checked-in fork can export the desired embeddings once its
affinity path runs, but it cannot trigger that path from these PPI YAMLs without
an additional protein-binder patch. `--affinity-side` / `--binder-override`
exist only for such a patched runner. `1AO7_ABC_DE` additionally needs a true
multi-chain group binder; choosing D or E alone is not scientifically neutral.
The correct pooling semantics are binder group versus the other measured
partner, with binder tokens excluded from the receptor mask.

The completed production extraction bypassed that direct-path limitation. The
structure path ran for all 1,705 inputs and retained the trunk pair tensor `z`;
the tracked `scripts/_build_aff_emb.py` then pooled the intended interface and
applied the two affinity MLPs post hoc. The consolidated result is
`data/peptide_systems/modeling/features/boltz_embeddings.npz` with normalized
keys `ids`, `target`, `pair_mean`, `head_ens1`, `head_ens2`, and `head_mean`.
Effective auto-selected binders are 1A22=A, 1AO7=DE, 1BRS=D, 1CHO=I, 1GC1=C,
1JTG=B, 1VFB=C, 2B2X=A, 3BT1=A, 3HFM=Y, 3S9D=A, 3SE3=B, and 4G0N=B.

Companion TSVs in the same bundle are intentionally consolidated copies of the
generated tables: `labels.tsv` is the primary one-row-per-`input_id` modeling
table and exactly concatenates the 13 `variants.tsv` files; `measurements.tsv`
is the one-row-per-observation replicate table and exactly concatenates the 13
per-system `measurements.tsv` files; `manifest.tsv` is the top-level generated
system manifest; `index.tsv` is a human-readable NPZ row map derived from
`ids`. `measurements.tsv` is not a second experimental dataset.

The transferred modeling dataset passed these one-time validation checks:
all 1,705 finite, unique embedding IDs join exactly to 1,705 labels; every
system has one WT; `index.tsv` reproduces NPZ order; all 2,136 observations
recompute the stored replicate summaries; and all manifest counts recompute.
Repeated experimental rows are intentional provenance rather than duplicate
features: 292 mutation-label groups repeat, 280 have differing mutant Kd, 250
use differing WT references, and 286 have differing delta-delta-G. Only six
groups are identical across all affinity fields, with four additional aliases
that differ only in multi-mutation token order. Keep one feature row per
`input_id`, retain every observation, and aggregate labels on the free-energy
scale.

The dataset was produced with `_build_aff_emb.py`'s default compatibility mode,
which omits the affinity MLP's trailing ReLU. Thus its `head_*` values are
pre-final-ReLU (negative values are expected), not the fork writer's exact
post-ReLU `g_head`. Apply ReLU to each ensemble head separately before
averaging, or regenerate with `--final-relu`, for a faithful `g_head` pass.
The primary `pair_mean` representation is unaffected.

The small-molecule training domain is **the premise of this stress test**, not
an execution blocker: the goal is explicitly to learn whether those
out-of-domain affinity representations still encode protein-interface mutation
effects.

A normalized-schema producer is now present as `scripts/_build_aff_emb.py`, so
the active dataset's `pair_mean`/head construction is locally specified even
though the exact historical BH3/p53 producer is not needed for the active
direction. The saved trunk tensors, structures, confidence JSONs, checkpoint,
and full production run log remain on the execution machine rather than in the
modeling dataset. Raw Boltz scalar affinity outputs were not included, so the
dataset alone does not support the raw-scalar baseline or re-pooling with
another binder choice.

### Intended per-system modeling

Build one model per system, analogous to Part 1 but with continuous ΔΔG as the
primary task. Use one row per unique `input_id`, median ΔΔG as the primary label,
and retain replicate count/spread for uncertainty analysis. Compare a simple
mutation baseline against Boltz embeddings and their combination; later add
LRIP interaction-energy features by joining on `(system, input_id)`. The dataset
is ready for that combined modeling step once LRIP features exist, but it does
not contain the poses/MM-GBSA outputs needed to generate LRIP. Prefer
WT-difference embeddings and strongly regularized Ridge/PLS with nested CV
because n is 47–275 while p is large.
Headline metrics are out-of-fold Spearman and ΔΔG-sign agreement, with MAE and
Pearson secondary. A position-held-out split is a stricter follow-up for
overlapping multi-mutants.

The initial implementation is `python -m boltz2_aff.peptide_pipeline`. It
validates the NPZ/index/label/measurement joins, uses WT only to construct
difference embeddings, evaluates mutants with nested random-variant CV, and
writes tidy out-of-fold predictions/metrics plus final Ridge models. Default
views are mutation-only, `pair_mean`, mutation+`pair_mean`, `head_mean`, and
mutation+`head_mean`; `--label mean` provides the mean-label sensitivity pass.
Inner CV selects Ridge alpha by MAE. Position-held-out CV, PLS, censor-aware
regression, replicate weighting, and LRIP joins remain follow-ups.

For per-system linear Ridge with an intercept, subtracting the same WT vector
from every mutant is only an affine translation of raw embeddings. Keep the
explicit delta representation for interpretation and future nonlinear or
cross-system work, but do not claim raw-vs-delta Ridge as distinct signal.

The exact modeling-file contract and transfer validation record are in
`data/peptide_systems/modeling/README.md`; per-system counts are in its
`manifest.tsv`.

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
- Classification uses a `RandomForestClassifier(n_estimators=200,
  class_weight="balanced", max_features="sqrt")` with no PCA step.
  PCA was removed 2026-05-20 because it was too aggressive in small-n regimes
  (ROCK1 n=68 compressed to 13 components, DRD3 n=32 to 6), causing AUC to
  drop from ~0.908 to ~0.808 on ROCK1 vs the earlier LR baseline. RF's
  own `max_features="sqrt"` already handles high-dim implicitly.
- Regression uses `RidgeCV` over `np.logspace(-4, 4, 33)`. When
  `boltz_affinity_pred_value` is present in ≥80% of rows, the regressor
  trains on the **residual** `p_affinity − (−boltz_pred_value)` so it
  learns corrections on top of Boltz-2's own scalar rather than the raw
  affinity. CV predictions are back-transformed to absolute `p_affinity`
  before metrics are computed; `residual_mode` and `residual_boltz_column`
  are logged in `metrics_regression.json`.
- The pipeline always emits both `classifier.joblib` and `regressor.joblib`.
- Motivation for the RF switch: Ji et al. (IP-SF, JCIM 2021) and Niu et al.
  (LRIP-SF) both show linear classifiers perform worst for per-residue
  interaction-energy features; GBDT/RF consistently outperform by large
  margins. The same nonlinearity argument applies to learned embedding
  features.

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

```powershell
# Part-1 pipeline: one/several/all targets (omit --targets for all 10)
python -m boltz2_aff.pipeline --targets ROCK1 --out-dir runs/rock1_embeddings
python -m boltz2_aff.pipeline --out-dir runs/all_embeddings

# Restrict embedding components; compare scalar/scores/combined feature sets
python -m boltz2_aff.pipeline --embedding-keys pair_mean1 --out-dir runs/pm1
python -m boltz2_aff.pipeline --feature-set boltz        --out-dir runs/boltz_scalar
python -m boltz2_aff.pipeline --feature-set ulvsh_scores --out-dir runs/ulvsh_scores
python -m boltz2_aff.pipeline --feature-set combined     --out-dir runs/combined

# Embedding-component sweep + unbiased nested CV
python scripts/sweep_embedding_keys.py --feature-set combined --out-root runs/sweep_combined
python scripts/nested_cv.py --out runs/nested_cv.json

# LRIP (run with PYTHONPATH=.)
PYTHONPATH=. python scripts/model_lrip.py
PYTHONPATH=. python scripts/model_lrip_combined.py
```

## Part 1 Classification Results (2026-05-20)

All 10 targets have embeddings under
`data/ulvsh/modeling/features/boltz_embeddings/`; sweeps write to
`runs/sweep_embeddings/` and `runs/sweep_combined/` (RF classifier, no PCA — the
earlier RF+adaptive-PCA variants are superseded). `scripts/nested_cv.py` is the
definitive read: outer/inner StratifiedGroupKFold(3) with the inner loop
selecting the best of 9 embedding combos on the training split, plus fixed
`pair_mean1` on the same outer folds (apples-to-apples). Results in
`runs/nested_cv.json`.

| Target | nested | fixed_pm1 | B2C | n |
|--------|--------|-----------|-----|---|
| ADRA2B | 0.667 | 0.500 | 0.611 | 13 |
| CASR | 0.847 | 0.837 | 0.645 | 148 |
| CNR1 | 0.625 | 0.672 | 0.373 | 45 |
| CNR2 | 0.782 | 0.636 | 0.740 | 60 |
| DRD3 | 0.815 | 0.889 | 0.846 | 32 |
| DRD4 | 0.756 | 0.768 | 0.723 | 324 |
| MTR1A | 0.285 | 0.380 | 0.597 | 36 |
| ROCK1 | 0.881 | 0.896 | 0.854 | 68 |
| SC6A4 | 0.871 | 0.894 | 0.827 | 33 |
| SGMR2 | 0.798 | 0.803 | 0.795 | 205 |
| **Median** | **0.798** | **0.803** | **0.740** | |

**Honest reading:** the combined/embedding models
beat raw **B2-C** on 8/10 (median 0.798 vs 0.740), but B2-C is not the whole raw
story — against the *better* of B2-A/B2-C per target (median ~0.77) the aggregate
margin largely disappears. The learned embeddings ≈ raw Boltz-2, not a decisive
win. Removing adaptive PCA was the decisive modeling change (recovered ROCK1
0.808→~0.90). ROCK1 is a favorable, unrepresentative target; persistent failures
at tiny n are ADRA2B (n=13, noise) and MTR1A (n=36). There is no universal best
embedding component, but `pair_mean1` is a reasonable fixed default; inner-fold
selection adds value mainly on CNR2 (0.636→0.782).

Regression is effectively broken outside ROCK1:

- 4/10 targets (CNR1, DRD3, SC6A4, SGMR2) have **zero** uncensored numeric
  affinity rows — regression is skipped entirely.
- Of the 6 that fit, only ROCK1 (n=27) gives a strong positive Pearson
  (~0.70). CASR (n=148) is weakly positive (~0.50); DRD4 (~0.10) and
  ADRA2B/MTR1A/CNR2 are near-zero or actively negative.

**Why regression is hard here:** ULVSH only provides numeric Ki/IC50/Kd for
confirmed actives — inactives have censored (">10µM") or percent-inhibition
measurements. So regression trains on a narrow affinity window among
structurally similar actives, with no negative training signal. Classification
just needs to rank actives above inactives; regression must predict exact
affinities within 2-3 log units. Much harder.

**Per-target regression picture (combined feature set):**
- ROCK1: Pearson ~0.70, screening AUC ~0.84 — solid; residual mode helps.
- CASR: Pearson ~0.50 — modest but positive; p>>n (1055 features, 148 rows)
  is the main bottleneck.
- DRD4: Pearson ~0.07-0.11 — essentially useless; likely assay heterogeneity
  (Ki/IC50/Kd mixed from different experimental conditions).
- CNR2: *negative* Pearson — anti-correlated; small numeric subset with a
  structural confound (tight binders cluster in a region the embedding ranks
  low).
- ADRA2B, MTR1A: too noisy/negative to interpret.

**Screening AUC is the meaningful regression metric.** The `regression.cv_roc_auc`
(train on actives-only rows per fold, predict all test rows, AUC vs `active_bool`)
mirrors how the Rognan paper evaluates Boltz-2 scalars. ROCK1 reaches 0.84.
Pearson is reported but secondary.

**Options if regression improvement is desired:**
1. Focus on CASR only — enough data; try explicit PCA→Ridge to reduce p>>n.
2. Tobit regression for censored targets — treat ">10µM" as a right-censored
   observation (scikit-survival or custom likelihood). Would recover CNR1, DRD3,
   SC6A4, SGMR2. Non-trivial, not yet implemented.
3. Frame regression as a side result for ROCK1/CASR only; classification (AUC)
   is the headline, matching the Rognan paper's framing.

No universal best embedding component across targets, but `pair_mean1` is a
reasonable fixed choice and wins on ROCK1, DRD3, CASR, CNR1 as headline cases.

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
- Regression is only meaningful for ROCK1 (n=27, Pearson ~0.70) and CASR
  (n=148, Pearson ~0.50). DRD4 (n=74) gives Pearson ~0.10 — likely assay
  heterogeneity. CNR2 is anti-correlated. Use `regression.cv_roc_auc` (screening
  AUC) as the primary regression metric, not Pearson.

**Dead-end tried and removed (2026-05-20):** ECFP4 Morgan fingerprints
(2048-bit, from `poses.mol2`) as `ligand` / `combined_ligand` feature sets. On
ROCK1, `ligand` 0.786 / `combined_ligand` 0.799 — both below `combined` without
them and below raw B2-C. B2-C already encodes the ligand pharmacophore; 2048
noise dims only hurt. Feature sets and the rdkit dependency were removed.

## Possible Next Steps

Part 1 is closed. Live options if it is revisited:

- Regression stays a side result for ROCK1 (screening AUC ~0.84) and CASR only;
  if expanded, Tobit regression for censored inactives is the principled path
  (would recover the four all-censored targets). Screening AUC, not Pearson, is
  the metric.
- PLS or PCA→Ridge to use the `head` components under p≫n rather than discarding.
- Why do CASR/DRD3 embeddings beat raw Boltz but ADRA2B/MTR1A fail? May track
  refolding accuracy (paper discusses this).
- LRIP follow-ups: see "LRIP future work" below.

## LRIP Interaction-Profile Feature Set — Run 2026-07-15

A fifth feature block based on **ligand–residue interaction profiles
(LRIP / IP-SF)** from the Junmei Wang lab, now built and evaluated. Method
references:

- `papers/bbab054.pdf` — Ji et al., *Briefings in Bioinformatics* 22(5) 2021:
  original IP-SF. Features = per-residue ligand–receptor interaction energies
  from docking → minimization/MD → MM-GBSA free-energy decomposition. GBDT,
  per-target. Reported mean ROC AUC 0.87 across 6 targets vs Glide 0.71.
- `papers/aef2177_CombinedPDF_v1.pdf` — Niu et al., LRIP-SF: scaled-up
  evaluation (670 complexes, 16 targets), mapping-pose (MP) vs DOCK pose
  protocols, random forest, plus selectivity (JAK1/TYK2) and a global
  sensitivity-analysis hotspot framework.

Why it matters here: LRIP is a *physically motivated, mechanistically
interpretable, target-specific* feature — the natural counterpoint to Boltz-2's
*learned latent* embeddings.

**Data.** Per-target per-residue interaction-energy matrices (one `<TARGET>.dat`
per target, rows = compounds, columns = per-residue energies) landed via SFTP in
`_sftp_lrip/` and were moved to `data/ulvsh/modeling/features/lrip/`. See that
directory's `README.md` for the `.dat` format and join details. Row ids join
directly to `labels.tsv` on `(target, ligand_id)` (ROCK1's ULVSH ids are
themselves `mol_01…mol_69`); two quirks handled by the harness: DRD3 `1_20`→
`1_2_0`, and ROCK1 `mol_44` absent (LRIP failure). 0 unmatched rows across all
targets. The compound set per target is the Boltz WT-input subset (`n_input_wt`),
minus a handful of MM-GBSA failures.

**Harness.** `scripts/model_lrip.py` (standalone LRIP vs embeddings vs raw
Boltz) and `scripts/model_lrip_combined.py` (paired increments). Both reuse
`boltz2_aff.modeling.train_classifier` + `boltz_baseline_metrics`, so LRIP uses
the identical RF → StratifiedGroupKFold → `cv_roc_auc` methodology and is
compared on the *same rows*. Run with `PYTHONPATH=.`. Outputs in `runs/lrip/`
and `runs/lrip_combined/` (per-target metrics/predictions/models + `summary.*`).

**Result 1 — standalone LRIP is a negative.** Median classification AUC across
10 targets: LRIP **0.612** < embeddings 0.744 = B2-A 0.744 < B2-C 0.759 <
best-of-raw-Boltz 0.793. LRIP beats best raw Boltz on 1/10 (ROCK1), beats
embeddings on 3/10, clears the paper's >0.65 "acceptable" bar on 4/10 (DRD3,
ROCK1, SC6A4, SGMR2). ROCK1 is again the standout (0.869) — favorable target,
not representative. Worst on small-n / anti-correlated targets (CNR1 0.376,
MTR1A 0.487, ADRA2B n=13 0.528). The IP-SF/LRIP-SF papers' ~0.87 is on curated
poses/actives; ULVSH is a harder, imbalanced screen on small per-target n.

**Result 2 — LRIP carries real signal but is redundant with the embeddings.**
Paired increments on identical rows:
- `boltz` scalars → `boltz + lrip`: median AUC 0.666 → **0.699** (median delta
  **+0.057**, helps 7/10; DRD3 +0.21, MTR1A +0.18, SC6A4 +0.09). LRIP adds real
  binding information beyond the six raw Boltz scalars.
- `combined` (embeddings + Boltz scalars) → `combined + lrip`: 0.748 → **0.750**
  (median delta **+0.008**, helps 8/10 but within noise). Once the learned
  embeddings are present, LRIP adds essentially nothing — the embeddings already
  subsume the per-residue interaction signal LRIP computes explicitly.

Net reading: consistent with the rest of Part 1 — learned ≈ physics, neither
clearly wins, and here with a clean mechanistic interpretation (Boltz-2's
embeddings have already learned the interaction-hotspot information).

### LRIP future work

- **Is the +0.008 combined increment real or noise?** Paired
  permutation/bootstrap over the per-target deltas (or a sign test across the
  8/10-positive targets) to put a CI on the increment before claiming any
  complementarity. This is the honest gate on whether `combined+lrip` is worth
  reporting as anything but "no effect."
- **Integrate as a first-class feature set** if pursued further: a
  `score_lrip_<resid>` column block discovered by `features.py` with `lrip` /
  `combined_lrip` choices, rather than the standalone scripts. Low priority
  given Result 2.
- **Per-target only where LRIP helps.** DRD3, MTR1A, SC6A4, ROCK1 show the
  largest boltz→+lrip gains; if a mechanistic hotspot story is wanted, these are
  where LRIP's per-residue interpretability (which residues drive the classifier)
  would be most informative — tie back to the Niu et al. sensitivity-analysis
  hotspot framework.
- **Provenance gap:** the poses/MM-GBSA protocol behind the transferred `.dat`
  files is not captured in-repo (pose source, minimization, GB model, decomp
  settings). Record it if these numbers get published; needed to reproduce or
  extend LRIP to new compounds.
- **Part 2 extension.** LRIP per-residue energies extend cleanly to
  peptide–protein interfaces and are especially interpretable there; joinable to
  `data/peptide_systems/modeling/` by `(system, input_id)` once poses/MM-GBSA
  exist for that set (see Part 2 sections). Still blocked on generating LRIP for
  peptides — the transferred ULVSH `.dat` set does not cover it.

## Part 2: Peptide / Mutation Robustness — BH3 / p53 arm (on hold)

> **On hold as of 2026-06-29;** superseded by the active `data/peptide_systems/`
> SKEMPI set above. Kept because it may return. Data under `data/peptides/`.
> Full pre-condensation detail (per-file provenance, exact counts) is in git
> history.

**Goal.** Test whether Boltz-2 (raw scalars *and* the embedding models) tracks
the **effect of mutations** on binding, using peptide ligands with deep
mutational series — extending the Rognan receptor-side mutation/shuffle
challenge to the *ligand* side ("physics vs. memorization"). Evaluation is
**within-series Spearman** and **ΔΔG sign agreement** vs the WT peptide, not
pooled AUC. LRIP per-residue energies would be especially interpretable here.

**Systems and data** (`data/peptides/source/<system>/`, ingest scripts
`scripts/parse_bh3_sortcery.py` / `parse_p53_li2010.py`):

- **BH3 ↔ Mcl-1/Bcl-xL/Bfl-1 (headline).** Keating-lab SORTCERY apparent
  affinities (Jenson et al. PNAS 2018; repo `papers/peptides/bh3/sortcery_design/`).
  689 cross-target peptides (Bim/PUMA backgrounds) × 3 receptors.
- **p53 TAD ↔ MDM2/MDMX (secondary).** Li, Pazgier et al. JMB 2010 SPR
  Ala-scan of PMI + (17–28)p53; ~21 single point mutants/receptor. Tables
  encoded as Python literals (PDF tables are raster images).

**Cofolding inputs** (`scripts/make_boltz_inputs_{bh3,p53}.py` →
`data/peptides/boltz/inputs/<system>/<receptor>/input/<peptide_id>.yaml`): each
YAML is two protein chains (A=receptor, B=peptide) with `properties.affinity.binder: B`.
Total **2139 runs** (2067 BH3 + 72 p53) — the tractable starting scope; larger
SORTCERY sets remain behind the same generators.

**Mutation-injection design (decided 2026-05-17).** Option A — **re-cofold each
mutant** (full Boltz-2 per variant; both structure and affinity-head paths live)
is **PRIMARY and done**. Option B — fix the WT pose and vary only the affinity
input (isolates the affinity-head path) is a **follow-up diagnostic**, built only
if A reproduces the Rognan insensitivity. Architecture constraint: keep pose
generation and affinity scoring separable so B can be added without a rewrite.

**First results (2026-05-22, embedding-model arm only — no raw-Boltz scalar
baseline yet).** Outputs in `runs/peptide_embeddings/`; all use `head_mean`.

- **BH3** (n=689/receptor): supervised CV Spearman(embeddings→apparent affinity)
  0.66/0.77/0.79 (Bcl-xL/Mcl-1/Bfl-1). Holds under within-background CV; the one
  weak case is PUMA-background on Bcl-xL (0.27). Against the SORTCERY replicate
  **noise ceiling** (0.83/0.96/0.92) the model recovers ~79–86% of achievable
  ranking signal. Embeddings also capture Bcl-2-family **selectivity** (predicted
  vs measured receptor preference, Spearman 0.67–0.77).
- **p53** (too few point mutants for supervision): model-free magnitude probe
  Spearman(embedding shift-from-WT, |ΔΔG|) 0.80–0.92 (PMI), 0.65–0.72 (p53
  17–28). WT-anchored ΔΔG-sign agreement = **1.00 on the clear effects**
  (|ΔΔG| ≥ 1 kcal/mol) in 6/8 series.
- The embedding shift under mutation is *small* (cosine-to-WT ≥ 0.99, ~2–3.5% of
  ‖WT‖) — consistent with the Rognan insensitivity concern — but the mutational
  signal *is* present in the representation feeding Boltz-2's scalar heads.

**Open item:** the raw-Boltz scalar baseline (B2-A/B2-C within-series Spearman +
ΔΔG-sign, the direct Rognan comparison) is built and verified
(`scripts/part2_raw_boltz_baseline.py`) but blocked on running Boltz-2 over the
2139 input YAMLs so the affinity JSONs exist; re-running the script then produces
it. Analyses: `scripts/analyze_peptide_embeddings.py`, `part2_analysis.py`,
`part2_extras.py`.
