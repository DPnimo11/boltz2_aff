# Modeling context — SKEMPI ΔΔG × Boltz-2 embeddings (Part 2)

Purpose-written for the downstream modeling AI. This bundle has everything needed for the primary
**Boltz-2 affinity embeddings (features) + experimental ΔΔG (labels)** analysis on a curated
**SKEMPI subset of 13 protein–protein complexes**. Goal: does Boltz-2's representation track how
point/combinatorial mutations change binding? Full extraction provenance lives on yuan
(`/work/jwang/boltz2/peptide_systems/{README.md,AGENTS.md,clickff_run_log.md}`).

## Files in this bundle
| file | what |
|---|---|
| `affinity_embeddings.npz` | **features.** 1,705 structures. keys: `ids`, `target`, `pair_mean`, `head_ens1`, `head_ens2`, `head_mean`. |
| `index.tsv` | NPZ row → `system` → `input_id`; derived from NPZ `ids` and kept as a human-readable row map. |
| `labels.tsv` | **Primary modeling table:** one row per unique structure/`input_id`, including median ΔΔG and mutation metadata. Exact concatenation of the 13 generated `variants.tsv` files. |
| `measurements.tsv` | **Replicate table:** one row per experimental observation, mapped many-to-one to `input_id`. Exact concatenation of the 13 generated per-system files; use for uncertainty/noise analyses, not as additional feature rows. |
| `manifest.tsv` | **System table:** one row per complex with chain groups, WT reference ΔG, binder setting, and counts. Exact copy of the generated top-level manifest. |

## Verified state (2026-07-10)
- A one-time transfer audit verified that 1,705/1,705 unique feature IDs join to 1,705/1,705 unique labels; all 2,136 observations
  (2,123 mutant + 13 WT) reproduce the stored summary statistics; every system has one WT; there
  are no exact duplicate embedding rows; and all manifest counts recompute exactly.
- The UTF-8 punctuation in this file may display as mojibake in Windows PowerShell 5's default
  console encoding. The file bytes are valid UTF-8.

## Embedding schema (`affinity_embeddings.npz`)
- `ids`  (1705,)  string, format **`<system>::<input_id>`**, e.g. `1VFB_AB_C::1VFB_WT`.
- `target` (1705,) string = system (e.g. `1VFB_AB_C`).
- `pair_mean`  (1705, **128**) — pooled trunk pair representation `z` over the binder↔partner interface (pre-MLP).
- `head_ens1`, `head_ens2`, `head_mean` (1705, **384**) — the two affinity-head MLPs applied to `pair_mean`, and their mean.
- Row `k` of every array corresponds to `ids[k]`. `index.tsv` gives the same order as (row, system, input_id).
- **Prefer `pair_mean`** as the primary feature: the head MLPs are **small-molecule-trained** (out-of-domain for
  protein binders by design), so `pair_mean` is the cleaner signal. `head_*` are provided for comparison.
- **Head activation detail:** this archive was produced with `_build_aff_emb.py`'s default behavior,
  which omits the affinity MLP's trailing ReLU to remain compatible with the earlier normalized
  BH3/p53 artifacts. The negative values in `head_*` confirm that these are pre-final-ReLU outputs,
  not the fork writer's exact `g_head`. For a faithful `g_head` comparison, apply ReLU separately to
  `head_ens1` and `head_ens2` and then average, or regenerate with `--final-relu`. `pair_mean` is
  unaffected.

## Labels (`labels.tsv`, one row per `input_id`)
Key columns: `input_id`, `system`, `pdb_id`, `mutation`, `n_substitutions`, `n_measurements`, `measurement_ids`,
**`ddg_median_kcal_mol`** (primary label), `ddg_mean/sd/min/max_kcal_mol`, `dg_median_kcal_mol`,
`group1_chains`, `group2_chains`, `affinity_binder` (blank → auto), `chain_sequences_json`.

## How to join features ↔ labels
`ids[k]` splits on `"::"` into `(system, input_id)`. Match that `input_id` (within its `system`) to the
`labels.tsv` row. Each system has exactly one **WT** row (`mutation == "WT"`, `ddg_median == 0`).
**Do not concatenate by row position:** `index.tsv` is in exact NPZ order, but `labels.tsv` is not.

Future LRIP or other feature blocks should use the same `(system, input_id)` key and can then be
joined directly to `labels.tsv` and the embedding rows. The bundle is sufficient for that combined
model once those new features exist; it does not itself contain the poses or energy calculations
needed to generate LRIP.

## Conventions (do not violate)
- **ΔΔG sign:** `> 0` weakens binding, `< 0` strengthens. Primary label = **`ddg_median_kcal_mol`** (median on the
  free-energy scale). **Never average raw Kd** across replicates — median/mean on the ΔG scale only.
- One modeling row per unique `input_id` (structures are deduplicated; replicate *observations* are kept in
  `measurements.tsv`, not collapsed into features).

## Recommended analysis (from `peptide_systems_log.md`; modeling is user-owned)
1. **Per system** (13 separate models). Primary signal = **`embedding(mutant) − embedding(WT)`** (subtract the
   system's WT row) vs `ddg_median_kcal_mol`.
2. Compare: mutation-only baseline vs embeddings-only vs combined. Add LRIP/interaction-energy features if available.
3. **Strongly-regularized Ridge or PLS with nested CV** — each system has only 47–276 unique mutants, embeddings are high-dim.
4. Headline metric: **out-of-fold Spearman**; secondaries: ΔΔG sign-agreement, MAE, Pearson.
5. **CV leakage rules:** never split repeated measurements of one `input_id` across folds; run random-variant CV first,
   then **position-held-out CV** as a stricter leakage test for overlapping multi-mutants.
6. Use `measurements.tsv` replicate spread to estimate assay noise; run a median-vs-mean label sensitivity pass.

## Caveats
- **Embedding nondeterminism (~1%).** Boltz's trunk `z` is not bit-reproducible run-to-run (bf16 AMP +
  nondeterministic CUDA kernels): two identical re-runs of one input differ by ~1% (cosine ≈ 0.999). The
  `mutant − WT` difference inherits this. If it competes with signal, consider replicate-averaging or treat small
  differences as noise. (This is why validation uses cosine, not bit-exact equality.)
- **Detection-limit / censored ΔΔG** (weak-binding caps in `measurements.tsv`) are preserved but **not** inferred as
  censored. Flag/exclude them in a sensitivity pass, or use a censored likelihood. *Open modeling decision.*
- **Out-of-domain head:** applying the small-molecule affinity head to PPIs is an intentional stress test, not a bug.
- **Single-sequence structures:** inputs are `msa: empty`; 654/1,705 have ipTM < 0.5 (weaker interfaces, partly from
  destabilizing mutants). Confidence (ptm/iptm/plddt) remains in each `confidence_*.json` under the production
  machine's `_output/` tree if you want it as a QC filter or auxiliary feature; it is not in this bundle.

## Binder side (affects `pair_mean`)
The interface mask is asymmetric (binder–binder self-pairs kept, receptor–receptor not), so **which chain group is
"binder" changes the embedding**. Default used here = **smaller chain group by residue count** (auto; `manifest.affinity_binder`
was blank). Per-system: 1AO7 binder=DE, 1CHO=I, 1VFB=C, 2B2X=A, 3HFM=Y, others the single small chain. To change it,
set `manifest.affinity_binder` and re-run Stage B (needs the trunk `z`, which is kept on **clickff** only).

## Redundancy and scope
- `index.tsv` is a human-readable projection of NPZ `ids`; NPZ `target` is also derivable from the
  prefix of each ID. Both are intentionally retained for easy inspection and safer joins.
- `head_mean` is exactly `(head_ens1 + head_ens2) / 2` and costs about 2.42 MB compressed. It is the
  largest removable redundancy, but retaining it keeps the schema compatible with the earlier peptide artifacts.
- `labels.tsv`, `measurements.tsv`, and `manifest.tsv` repeat information in the generated tables
  under `data/peptide_systems/boltz_inputs/`. These are intentional consolidated copies: the
  per-system tree is gitignored and rebuildable, while this modeling directory is versioned and
  self-contained. In particular, `measurements.tsv` is not a second experimental dataset.
- Repeated experimental observations are intentional assay replicates or differently normalized
  measurements. They are not duplicate feature rows and must not be deleted or split across folds.
- The bundle does **not** contain the saved trunk `z`, predicted structures, confidence JSONs, the
  Boltz checkpoint, raw Boltz scalar affinity outputs, or the complete production run log. Those are
  unnecessary for the primary embedding-versus-ΔΔG modeling, but are required respectively for
  re-pooling/re-extraction, structure-confidence QC, exact end-to-end reproduction, or a raw-scalar baseline.
