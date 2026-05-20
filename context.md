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

Candidate systems (all have ≥30 quantified mutational variants of one peptide
against one receptor; see "Suggested Peptide Systems" in README for dataset
pointers):

1. **BH3 peptide ↔ MCL-1 / BCL-xL** (Keating lab; Jenson et al. eLife 2018
   and related). Hundreds of α-helical ~26-mer BH3 variants (BIM/BID/PUMA
   backgrounds) with SPR/FP Kd against both MCL-1 and BCL-xL. Best dynamic
   range and the cleanest single + combinatorial mutant series. **Top pick.**
2. **p53 TAD peptide ↔ MDM2 / MDMX**. The p53(17–28) helix plus phage-
   optimized variants (pDI, PMI) and systematic single mutants — dozens of
   Kd values, well-curated, small well-defined interface.
3. **HLA-A*02:01 ↔ nonamer peptide** (e.g., CMV pp65 NLVPMVATV or influenza
   GILGFVFTL). IEDB / NetMHCpan give hundreds-to-thousands of single-
   substitution variants with measured IC50 — huge n, but mixed assay
   conditions; good for a scale stress-test.
4. **PDZ domain (PSD-95 PDZ3) ↔ CRIPT C-terminal peptide**. Peptide-side
   saturation data; short interface, lower dynamic range — use as a harder
   secondary case.

Caveats to design for: assay-type heterogeneity (Kd vs IC50 vs FP), narrow
affinity dynamic range for some peptide series, and the fact that Boltz-2's
affinity head was trained predominantly on small molecules — peptide
generalization is exactly the open question, not an assumption.

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
