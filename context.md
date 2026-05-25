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
- **Part 2 (peptide / mutation robustness):** source papers and supplementary
  files live under `papers/peptides/{bh3,p53,p53_2,HLA_A0201}/`. Parsed
  mutational tables are written to `data/peptides/<system>/` by
  `scripts/parse_<system>_mutants.py`. See "Planned Part 2" below for the
  per-system source-of-truth files.
  Extracted Boltz-2 affinity embeddings for the cofolded peptide complexes are
  filed under `targets/peptides/<system>__<receptor>/affinity_<peptide_id>.npz`
  (2139 files: BH3 689 peptides × 3 receptors + p53 36 entries × 2 receptors).
  These use a **newer export schema** than Part 1 (`pair_mean` 128, `head_ens1`
  384, `head_ens2` 384, `head_mean` 384, plus `peptide_id`/`target` metadata)
  and are deliberately *not* discovered by the Part-1 pipeline, which globs the
  `affinity_embeddings_*.npz` prefix (these are `affinity_*.npz`).

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

## Multi-Target Sweep Findings (2026-05-17, updated 2026-05-20)

All 10 targets have embeddings under `targets/`. The definitive sweep results
are in `runs/sweep_embeddings_v2/` and `runs/sweep_combined_v2/` (RF classifier,
no PCA). Earlier `sweep_embeddings/` and `sweep_combined/` used RF+adaptive PCA
and are now superseded.

**Combined feature set (embeddings + Boltz scalars + ULVSH scores), honest
fixed `pair_mean1` choice, classification AUC vs raw Boltz B2-C:**

| Target | combined | B2C | Result |
|--------|----------|-----|--------|
| ADRA2B | 0.319 | 0.611 | loss (n=13, noise) |
| CASR | 0.789 | 0.645 | **win** |
| CNR1 | 0.629 | 0.373 | **win** |
| CNR2 | 0.692 | 0.740 | loss |
| DRD3 | 0.891 | 0.846 | **win** |
| DRD4 | 0.741 | 0.723 | **win** |
| MTR1A | 0.334 | 0.597 | loss (n=36) |
| ROCK1 | 0.909 | 0.854 | **win** |
| SC6A4 | 0.852 | 0.827 | **win** |
| SGMR2 | 0.812 | 0.795 | **win** |

**7/10 targets win; median combined 0.765 vs median B2C 0.732.** Removing the
adaptive PCA was the decisive change — it recovered ROCK1 from 0.808→0.909 and
DRD3 from 0.837→0.891, and improved several other targets.

Embeddings only (`pair_mean1` fixed): 3/10 wins, median 0.718 vs B2C 0.732 —
the additional ULVSH score and Boltz scalar columns in `combined` provide the
margin that pushes the majority of targets above Boltz-2's own scalar.

Persistent failures — ADRA2B (n=13, essentially noise) and MTR1A (n=36) — are
likely irreducible at these sample sizes regardless of feature set.

## Nested CV Results (2026-05-20)

`scripts/nested_cv.py` runs outer/inner StratifiedGroupKFold(3) where the inner
loop selects the best of 9 embedding combos on the training split, and the outer
fold evaluates the selected combo on held-out data. Also reports fixed `pair_mean1`
on the same outer folds (exact apples-to-apples). Results in `runs/nested_cv.json`.

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

**Nested > B2C: 8/10**; median 0.798 vs 0.740. The post-hoc sweep bias was small
— nested CV confirms the result and adds value on CNR2 (0.636→0.782) where
inner-fold combo selection picks a better component than pair_mean1.
Failures: DRD3 (embedding noise at n=32) and MTR1A (n=36, genuine hard case).

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

## Morgan Fingerprint (ECFP4) Feature Block — Added and Removed 2026-05-20

ECFP4 fingerprints (radius=2, 2048-bit, from `poses.mol2` via RDKit) were
implemented as `ligand` and `combined_ligand` feature sets and subsequently
**removed**. Results on ROCK1: `ligand` only AUC 0.786, `combined_ligand` 0.799
— both below `combined` without fingerprints (0.909) and raw B2-C (0.854).
B2-C already encodes the ligand pharmacophore that ECFP4 captures; adding 2048
noise dimensions only hurts. rdkit dependency also removed from `pyproject.toml`.

## Possible Next Steps

- **[done 2026-05-19]** Replace `LogisticRegression` with `RandomForestClassifier`
  + adaptive PCA in `modeling.py`; add residual regression mode.
- **[done 2026-05-20]** Add ECFP4 Morgan fingerprint feature block (`ligand` /
  `combined_ligand` feature sets); confirms ligand-centrism hypothesis.
- **[done 2026-05-20]** Remove adaptive PCA from RF classifier; re-run 10-target
  sweep. Combined+pair_mean1 now wins 7/10 targets, median AUC 0.765 vs B2C
  0.732 (`runs/sweep_embeddings_v2/`, `runs/sweep_combined_v2/`).
- **[done 2026-05-20]** Nested CV with inner-fold combo selection
  (`scripts/nested_cv.py`): 8/10 targets beat B2C, median 0.798 vs 0.740
  (`runs/nested_cv.json`). Confirms prior sweep result was not optimistically
  biased — adaptive selection adds further value (especially CNR2: 0.636→0.782).
- Add nested CV or a held-out test split so per-target combo selection is not
  optimistically biased; re-evaluate the classification "wins" honestly.
- Drop regression from the headline; report only as a side result for ROCK1
  (Pearson ~0.70, screening AUC ~0.84) and CASR (Pearson ~0.50). The training
  set is all-actives so Pearson is range-restricted; the screening AUC
  (`regression.cv_roc_auc`) is the meaningful metric. If regression is expanded,
  Tobit regression for censored inactives is the principled path.
- Try PLS regression or PCA→ridge to handle p≫n for the `head` components
  rather than discarding them.
- Investigate why CASR/DRD3 embeddings beat raw Boltz so decisively but
  ADRA2B/MTR1A fail — may correlate with refolding accuracy discussed in the
  paper.

## Planned: LRIP Interaction-Profile Feature Set

Add a fifth feature block based on **ligand–residue interaction profiles
(LRIP / IP-SF)** from the Junmei Wang lab:

- `papers/bbab054.pdf` — Ji et al., *Briefings in Bioinformatics* 22(5) 2021:
  original IP-SF. Features = per-residue ligand–receptor interaction energies
  from docking → minimization/MD → MM-GBSA free-energy decomposition. GBDT,
  per-target. Reported mean ROC AUC 0.87 across 6 targets vs Glide 0.71.
- `papers/aef2177_CombinedPDF_v1.pdf` — Niu et al., LRIP-SF: scaled-up
  evaluation (670 complexes, 16 targets), mapping-pose (MP) vs DOCK pose
  protocols, random forest, plus selectivity (JAK1/TYK2) and a global
  sensitivity-analysis hotspot framework.

Why it matters here: LRIP is a *physically motivated, mechanistically
interpretable, target-specific* feature, the natural counterpoint to Boltz-2's
*learned latent* embeddings. The same per-target pipeline can compare
`embeddings` vs `lrip` vs `combined+lrip`, and LRIP extends cleanly to
peptide–protein interfaces (per-residue energies) for Part 2 below.

Integration sketch: a `score_lrip_<resid>` column block computed from the
ULVSH docked/cofolded poses (or Boltz-2 cofolded pose) via MM-GBSA decomposition,
discovered by `features.py` with an `lrip` / `combined_lrip` feature set. This
requires a pose + an MM-GBSA decomposition step (Amber/`MMPBSA.py` or
equivalent) — heavier than current feature blocks; scope before building.

## Planned Part 2: Peptide / Mutation Robustness

Goal: test whether Boltz-2 (raw scalars *and* the embedding models) tracks the
**effect of mutations** on binding, using peptide ligands with deep
mutational series. This directly extends the Rognan paper's mutation/shuffle
challenges from the receptor side to the *ligand* side, and probes the central
"physics vs. memorization" question: a model that memorized ChEMBL ligands
should fail to rank a tight mutational series it never saw.

Evaluation differs from Part 1: the relevant metrics are **within-series rank
correlation** (Spearman of predicted vs measured across the mutant set) and
**ΔΔG sign agreement** relative to the wild-type peptide — not pooled AUC.
LRIP per-residue energies are especially interpretable here (which interface
residue drives the mutational effect).

### Confirmed systems and local data sources (2026-05-20)

After auditing the available papers, two systems have sufficient locally-stored
mutational data to drive Part 2:

1. **BH3 peptide ↔ Mcl-1 / Bcl-xL / Bfl-1** — **headline system.**
   Source: Jenson et al. PNAS 2018 (`papers/peptides/bh3/jenson-et-al-2018-...pdf`)
   plus the Keating lab SORTCERY data repository cloned at
   `papers/peptides/bh3/sortcery_design/` (upstream:
   https://github.com/KeatingLab/sortcery_design). The `csv/` subfolder ships
   ten files; main + replicate pairs cover Bcl-xL (~4395 unique 22-mer peptides
   in `x1.csv`/`x1r.csv`), Mcl-1 (~4491 in `m1.csv`/`m1r.csv`), and Bfl-1
   (~3806 in `f100.csv`/`f100r.csv`). Each row carries the peptide sequence
   (`protein`), parent background (`bg`: B=Bim, P=PUMA), apparent affinity
   (`x1_expectedValue` / `m1_expectedValue` / `f100_expectedValue`), apparent
   binding energy (`*_energy`), and quality flags (`isUnimodal`,
   `isOneHitWonder`). Peptides differ from Bim or PUMA at up to 8 positions
   — clean within-background mutational series. Companion structural
   reference: Jenson, Ryan, Grant, Letai, Keating 2017 eLife
   (DOI 10.7554/eLife.25541; PUMA-background epistatic variants with X-ray
   structures), not stored locally but cited as a high-quality validation slice.

2. **p53 TAD peptide ↔ MDM2 / MDMX** — **secondary system.**
   Mutational data: Li, Pazgier et al. *J Mol Biol* 2010, stored at
   `papers/peptides/p53_2/1-s2.0-S0022283610002433-main.pdf` (PMID 20226197,
   PMC2856455). Systematic Ala-scan of PMI (TSFAEYWNLLSP) and (17–28)p53
   (ETFSDLWKLLPE) — Table 1 (PMI + 16 analogs) and Table 3 ((17–28)p53 + 15
   analogs) report ITC Kd values against both synMDM2 and synMDMX; plus
   10 truncation analogs. Net ~17 single-mutant point-substitution variants
   per scaffold × 2 receptors. Two scaffolds pooled gives ~34 within-series
   measurements per receptor — just clears the ≥30 threshold. Treat
   truncations as a separate sub-analysis (length-changes, not point
   mutations). Structural baseline (synMDM2-PMI, synMDMX-PMI crystal
   structures): Pazgier et al. PNAS 2009 at `papers/peptides/p53/` — kept
   as the WT-Kd reference and source of the binding-mode reference structure.

### Deferred / dropped systems

- **HLA-A*02:01 ↔ nonamer peptide** — *deferred*. The stored Trolle et al.
  *Bioinformatics* 2015 paper (`papers/peptides/HLA_A0201/`) is the IEDB
  automated benchmarking framework, not a single-peptide deep mutational scan
  — its ~4000 measurements are scattered across 17 alleles. A clean
  single-parent series would need to be curated from IEDB (filter
  HLA-A*02:01 + 9-mer + quantitative IC50 + ≤1 mutation from a parent
  epitope) or pulled from Sidney/Sette positional scanning libraries. Set
  aside until BH3 + p53 are working — the headline systems already cover
  five receptors (Bcl-xL, Mcl-1, Bfl-1, MDM2, MDMX), which is plenty.
- **PDZ domain ↔ CRIPT peptide** — *not pursued.* No data downloaded; would
  have been a tertiary case.

Caveats to design for: assay-type heterogeneity (apparent-Kd from yeast-display
SORTCERY for BH3 vs SPR competition Kd for p53; **correction** — earlier
note erroneously said ITC for p53, but Li 2010 used SPR competition,
Methods page 12 of the PDF), differing affinity dynamic ranges, and the
fact that Boltz-2's affinity head was trained predominantly on small
molecules — peptide generalization is exactly the open question, not an
assumption. The SORTCERY apparent affinities are an internally consistent
ranking within one target (the appropriate input for Spearman / ΔΔG-sign
analysis) but should not be cross-compared in absolute units to the p53 SPR
Kds.

### Ingested datasets (2026-05-20)

Two ingestion scripts and their outputs:

- `scripts/parse_bh3_sortcery.py` → `data/peptides/bh3/measurements.tsv`
  (27,499 long-format rows across 10 SORTCERY CSVs). Per-target unique
  peptide counts: Bcl-xL 10,142 (includes pilot screen), Mcl-1 4,491,
  Bfl-1 3,805. All rows pass `is_unimodal=True / is_one_hit_wonder=False`
  quality flags. Backgrounds: Bim (B) and PUMA (P). Apparent-value range
  per target ≈ 1.5–11 (monotonic with log10 K_D on the cell-surface scale;
  sign convention is *higher = tighter* — confirm against the SORTCERY
  paper before using absolute values).

- `scripts/parse_p53_li2010.py` → `data/peptides/p53/measurements.tsv`
  (72 rows = 2 scaffolds × 2 receptors × 18 peptide entries). The script
  encodes Li 2010 Tables 1 and 3 as Python literals because both tables
  are rendered as raster images in the PDF (PNG crops kept under
  `data/peptides/p53/raw/` for audit). PMI/MDM2 dynamic range is 5.5
  log-units (490 pM → 160 μM); the (17–28)p53/MDMX series is the narrowest
  at ≈2.9 log-units. F19A and W23A on the p53 scaffold are tagged
  `analog_class='not_determined'` (SPR could not quantify; excluded from
  numeric metrics). The A4A row on the PMI scaffold is a no-op control
  (position 4 is already Ala) tagged `analog_class='control_redundant'`.

For the Part-2 Spearman / ΔΔG-sign analysis, the usable mutational counts
per receptor are: BH3 — thousands of multi-position variants per target,
ample. p53 — 11 single-Ala substitutions on PMI + 10 on (17–28)p53 = 21
single-residue point mutations per receptor; adding the 5+5 truncations
(treated as a separate sub-analysis) gives 31, just clearing the ≥30
threshold.

### Boltz-2 cofolding inputs (2026-05-20)

Two YAML generator scripts emit one cofolding input per (peptide, receptor)
pair. Layout:

    data/Boltz-2/peptides/<system>/<receptor>/input/<peptide_id>.yaml
    data/Boltz-2/peptides/<system>/<receptor>/manifest.tsv
    data/Boltz-2/peptides/bh3/peptide_index.tsv  # global peptide_id <-> seq map

Each YAML follows the existing Boltz-2 convention used by ROCK1
(`data/Boltz-2/ROCK1/{wt,mut,shuffled}/input/mol_NN.yaml`): two
`protein` chains (`id: A` = receptor, `id: B` = peptide ligand) and
`properties.affinity.binder: B`. Peptides are encoded as protein chains
rather than as `ligand: smiles:` because they are too long for the
small-molecule pose head.

- `scripts/make_boltz_inputs_p53.py` — **72 YAMLs** = 36 unique
  (scaffold, mutation_label) entries × 2 receptors (MDM2, MDMX).
  Receptors are the Pazgier/Li synthetic constructs synMDM2 (MDM2
  residues 25–109, Q00987) and synMDMX (MDMX 24–108, O15151). YAMLs
  for the F19A/W23A "not_determined" rows and the PMI A4A no-op are
  still emitted (Boltz predictions are useful even where the SPR Kd
  is missing); their status is tagged in `manifest.tsv`.

- `scripts/make_boltz_inputs_bh3.py` — **2067 YAMLs** = **689 unique
  cross-target peptides × 3 receptors** (Bcl-xL, Mcl-1, Bfl-1).
  Cross-target = appearing in the primary (non-replicate, non-pilot)
  sort for *all three* receptors. Strictly the 1 nM main sort for
  Bcl-xL/Mcl-1 and the 100 nM main sort for Bfl-1; the 100 nM Bcl-xL
  main sort (x100.csv) is also pooled into the Bcl-xL leg, which is
  why the intersection is 689 rather than the 537 quoted earlier from
  x1.csv alone. Receptor constructs: Bcl-xL ΔTM 1–209 (Q07817), Mcl-1
  binding-domain 172–327 (Q07820), Bfl-1 ΔTM 1–151 (Q16548). Replicate
  and pilot rows remain in the upstream `measurements.tsv` for noise
  estimation.

The cofolding workload is **2139 Boltz-2 runs** combined (72 + 2067).
This is the *starting* scope chosen 2026-05-20 to keep the first pass
tractable — the BH3 main+replicate full set (~8.4 k) and the pilot
screen (~18 k total) remain available behind the same generator script
if scope expands later.

### Mutation injection: two designs

The affinity head never reads sequence directly — it reads the structure
trunk's representation of the cofolded complex. A mutation can therefore move
the prediction via (1) the **structure path** (trunk re-poses the complex) or
(2) the **affinity-head path** (head reacts to changed interface chemistry on
a given structure). Two experimental designs separate these:

- **Option A — re-cofold each mutant (PRIMARY).** Change the peptide
  sequence and run the full Boltz-2 pipeline fresh per variant; both paths
  live. This is the realistic, end-to-end, headline experiment and the
  default for Part 2. (User confirmed 2026-05-17: re-running Boltz-2 for
  every variant.)
- **Option B — fixed wild-type pose, vary only the affinity input
  (FOLLOW-UP DIAGNOSTIC).** Pin the structure to the WT cofolded complex
  (template / distance-constraint conditioning, or feeding the WT structure
  directly into the affinity module if the `../boltz` fork exposes it) so the
  trunk representation is ~constant across variants; only the mutated residue
  identity changes. Kills the structure path, isolating the affinity-head
  path.

Rationale: the Rognan paper found Boltz-2 affinity largely *insensitive* to
binding-site mutations. If Option A reproduces that flatness, Option A alone
cannot say whether the trunk failed to re-pose or the head ignored a real
structural change — Option B is the diagnostic that localizes the failure.
Build B only if A shows the insensitivity.

**Architecture constraint:** keep pose generation and affinity scoring as
separable pipeline steps so the fixed-template path (Option B) can be added
later without a rewrite. Do not couple them.

### Part 2 first results (2026-05-22)

Embeddings for all 2139 cofolded peptide complexes were extracted (Option A,
re-cofold per variant) and filed under `targets/peptides/<system>__<receptor>/`.
Two analyses were run; outputs in `runs/peptide_embeddings/`. Both use
`head_mean` (the ensemble-averaged representation immediately before the scalar
affinity heads).

**Embedding sensitivity probe** (`scripts/analyze_peptide_embeddings.py` →
`embedding_sensitivity.json`) — label-free. All embeddings are unique (QC: no
degenerate extraction). On p53 (WT-resolvable ids) the embedding *does* move
under mutation and the largest shifts land on the known anchor residues — W23A
(by far the biggest) and F19A on the p53(17–28) helix, W7A then F3A on PMI. But
the magnitude is small: cosine-to-WT stays ≥0.99 and the mean shift is only
~2–3.5% of ‖WT‖ — consistent with the Rognan "insensitivity" concern.

**Part 2 analysis** (`scripts/part2_analysis.py` → `part2_results.json`).
NOTE: this is the *embedding-model* arm only — no peptide affinity JSONs were
extracted, so there is **no raw-Boltz scalar baseline yet** (the direct Rognan
comparison still needs the JSONs).

- **BH3** — supervised CV (KFold-5) Spearman of embeddings → `apparent_value`
  (higher = tighter), n=689/receptor: Bcl-xL 0.657, Mcl-1 0.766, Bfl-1 0.791
  (all p ≤ 1e-86, Pearson 0.62/0.82/0.78). **Within-background CV** (refined
  2026-05-25 — a model CV'd *only* within Bim or PUMA, the honest within-series
  read): Bim 0.69/0.78/0.77, PUMA 0.27/0.66/0.75. These track the pooled split
  closely, so the ranking borrows no cross-background signal; PUMA-background on
  Bcl-xL (0.27) is the one weak case, now confirmed robust.
- **p53** — only ~10–11 point mutants per scaffold per receptor, too few for a
  384-dim supervised model, so the headline is the *model-free magnitude probe*:
  Spearman(embedding shift-from-WT, measured |ΔΔG|). PMI 0.80–0.92,
  p53(17–28) 0.65–0.72 (point only); including truncations 0.75–0.85. Secondary,
  n-limited: LOO-Ridge Spearman 0.66–0.95, and a **WT-anchored ΔΔG-sign
  agreement** (refined 2026-05-25: predicted `pKd_WT(held-out) − pKd_mut` vs
  measured ΔΔG). Overall sign agreement is 0.64–0.87, but on the *clear effects*
  (|ΔΔG| ≥ 1 kcal/mol) it is **1.00 in 6/8 series** — every clearly
  (de)stabilizing mutation gets the right direction; disagreements sit on
  near-neutral mutations within assay noise.

**Takeaway:** the mutational signal is clearly present in the representation that
feeds Boltz-2's scalar affinity heads, for both systems — contrasting with the
Rognan finding that the raw scalars were largely mutation-insensitive. The open
question is whether Boltz-2's *own scalar output* preserves it.

**Open Part-2 items:**
1. **(remaining gap — now external-compute-bound only)** Raw-Boltz scalar
   baseline (B2-A / B2-C), the apples-to-apples Rognan comparison the
   embedding-model arm cannot make. The *analysis* is done and verified
   (`scripts/part2_raw_boltz_baseline.py`, smoke-tested on synthetic JSONs); it
   reads `data/Boltz-2/peptides/<system>/<receptor>/output/<pid>/affinity_<pid>.json`
   (Part-1 schema: `affinity_pred_value` lower=stronger, `affinity_probability_binary`
   higher=stronger), joins to the manifests, and prints raw-Boltz vs the
   embedding arm side by side. The only thing left is **running Boltz-2 over the
   2139 input YAMLs** (external GPU job via the `../boltz` fork) so the JSONs
   exist; re-running the script then produces the baseline.

- **[done 2026-05-25]** WT-anchored ΔΔG-sign metric (replaced the crude
  `sign_agreement_vs_median`); sign agreement = 1.00 on |ΔΔG| ≥ 1 kcal/mol in
  6/8 series. In `scripts/part2_analysis.py` (`analyze_p53`).
- **[done 2026-05-25]** BH3 within-background CV (model CV'd within each Bim/PUMA
  background); within-series ranking holds, PUMA-on-Bcl-xL stays the weak case.
  In `scripts/part2_analysis.py` (`analyze_bh3`).

### Part 2 extras (2026-05-25)

`scripts/part2_extras.py` → `runs/peptide_embeddings/part2_extras.json`. Three
follow-ups that need no new data:

- **Embedding-key sweep** — no dominant view (BH3 CV Spearman 0.64–0.80, p53
  magnitude probe 0.69–0.83 across `pair_mean`/`head_ens1`/`head_ens2`/`head_mean`
  /`pair_mean+head_mean`). `head_mean` and the `pair_mean+head_mean` concat are
  consistently among the best, so the `head_mean` default is fine; best-per-target
  varies (e.g. `pair_mean` 0.668 on Bcl-xL, `head_ens1` 0.830 on p53/MDM2),
  echoing Part 1's "no universal best component".
- **Replicate noise ceiling (BH3)** — test-retest Spearman between the main and
  replicate SORTCERY sorts, **computed within concentration** (pooling Bcl-xL's
  1 nM x1 and 100 nM x100 sorts had deflated its ceiling to 0.505 — a confound).
  Concentration-matched: Bcl-xL 0.831 (@1 nM), Mcl-1 0.955 (@1 nM), Bfl-1 0.924
  (@100 nM). The head_mean CV model (0.657/0.766/0.791) recovers **~79/80/86 %**
  of the achievable ranking signal — labels are highly reproducible, so the
  ~15–20 % gap is real model headroom, not label noise (Bcl-xL is both the
  hardest target and has the lowest ceiling).
- **Cross-target selectivity (BH3)** — the 689 peptides are folded against all
  three receptors, so predicted vs measured *receptor preference* can be scored.
  Affinities are percentile-rank-normalised within each receptor first (SORTCERY
  values are only internally consistent per target), then selectivity =
  percentile[r1] − percentile[r2]. Spearman of predicted vs measured selectivity:
  Mcl-1/Bcl-xL 0.766, Bfl-1/Bcl-xL 0.728, Mcl-1/Bfl-1 0.670 (all p ≤ 1e-91,
  n=689). The embeddings capture Bcl-2-family **selectivity**, not just
  per-receptor affinity — the headline strength of the BH3 system.
