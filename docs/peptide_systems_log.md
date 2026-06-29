# Peptide-system pipeline log

Last updated: 2026-06-29

This is the implementation and issue log for the active Part-2 SKEMPI subset.
The authoritative operational summary also lives in `AGENTS.md`; this file keeps
the audit details and unresolved decisions visible as the workflow evolves.

## Current state

- The 13 source bundles were moved from repository-root `peptide_systems/` to
  `data/peptide_systems/systems/<PDB>/`.
- Each PDB now has its own directory containing the PDB, FASTA, mapping, raw
  text table, curated `_New.txt` source of truth, Excel mirror, and optional
  `.wt` file.
- `scripts/make_boltz_inputs_peptide_systems.py` is the preserved generator.
- Generated inputs and manifests live under
  `data/peptide_systems/boltz_inputs/<system>/` rather than the Part-1
  `data/Boltz-2/` tree.
- Regeneration command:

  ```powershell
  python scripts/make_boltz_inputs_peptide_systems.py
  ```

- The generated `boltz_inputs/` products remain ignored and are recreated
  deterministically. `.gitignore` now explicitly exposes `systems/**` so future
  source bundles do not disappear behind the repository's data ignore rules.

## Dataset and generation audit

The curated tables contain 2,123 mutant measurement rows. Canonicalizing the
mutation tokens (including token order in multi-mutants) gives 1,692 unique
mutant complexes. Adding one WT per system gives 1,705 generated YAMLs.

| System | Measurement rows | Unique mutants | Inputs with WT | Repeat rows |
|---|---:|---:|---:|---:|
| 1A22_A_B | 251 | 179 | 180 | 72 |
| 1AO7_ABC_DE | 212 | 151 | 152 | 61 |
| 1BRS_A_D | 94 | 75 | 76 | 19 |
| 1CHO_EFG_I | 295 | 275 | 276 | 20 |
| 1GC1_G_C | 67 | 67 | 68 | 0 |
| 1JTG_A_B | 275 | 191 | 192 | 84 |
| 1VFB_AB_C | 72 | 63 | 64 | 9 |
| 2B2X_HL_A | 115 | 94 | 95 | 21 |
| 3BT1_A_U | 245 | 240 | 241 | 5 |
| 3HFM_HL_Y | 122 | 97 | 98 | 25 |
| 3S9D_A_B | 218 | 141 | 142 | 77 |
| 3SE3_B_A | 67 | 47 | 48 | 20 |
| 4G0N_A_B | 90 | 72 | 73 | 18 |
| **Total** | **2,123** | **1,692** | **1,705** | **431** |

There are 620 unique multi-mutants; the largest variant contains 15
substitutions. The generator validates every mutation's chain, source amino
acid, and coordinate before writing a YAML. It also verifies every curated
`DG = DG_WTref + (DGmut - DGWT)` value within rounding tolerance.

Each system output contains:

- `input/<input_id>.yaml` — one WT or unique canonical mutant complex.
- `variants.tsv` — one row per YAML, including replicate count, original row
  IDs, median/mean/range/SD of measured delta-delta-G, partner chains, and the
  exact generated chain sequences.
- `measurements.tsv` — one row per curated experimental observation, mapped to
  the shared `input_id`; no repeated affinity measurement is discarded.
- Top-level `boltz_inputs/manifest.tsv` — generation counts and chain metadata
  for all systems.

All 1,705 YAMLs were parsed successfully with PyYAML after generation. A second
generation produced the same counts and replaced no valid inputs.

## Resolved data issues

### Mutation numbering and 3S9D

Mutation numbers match `.mapping` field four, `SEQIDX`; they do **not** match
field three, `PDBNUM`. All 3,711 substitution tokens resolve to the expected
wild-type residue through `SEQIDX`.

For 12 systems, the mapping residue sequence exactly equals the FASTA and
`SEQIDX` is directly the FASTA position. `3S9D` is the exception: FASTA chains
A and B contain respectively two and five additional residues relative to the
mapping sequence. The generator therefore aligns each mapping sequence to its
FASTA and translates `SEQIDX` to the aligned FASTA coordinate. Directly using
the mutation number as a FASTA index would mutate the wrong residue in 3S9D.

### Repeated mutants

Repeated rows are predominantly independent or differently normalized
measurements, not accidental duplicates:

- 292 exact mutation-label groups occur more than once.
- 280 have different mutant Kd values.
- 250 use different WT Kd references.
- 286 have different delta-delta-G values.
- Only six are identical across all affinity fields.
- Four additional aliases differ only in multi-mutation token order.

Consequently, structure generation is deduplicated but experimental rows are
not. The primary modeling label should be median delta-delta-G per unique
mutant, with all observations retained for uncertainty and sensitivity
analysis. Do not average raw Kd values; average or take the median on the free-
energy/log scale.

### Measured chain groups

Only chains named by the curated filename's `<group1>_<group2>` partition are
included. `3SE3.fasta` also contains chain C, but the measured system is
`3SE3_B_A`, so C is excluded and recorded in the manifests. Revisit this choice
if the intended experiment requires the complete ternary complex rather than
the measured B-A pair.

## Open execution issues

1. **Protein affinity is not supported by stock Boltz-2.** The local
   `../boltz` parser requires `properties.affinity.binder` to name one
   small-molecule ligand chain. These inputs are protein-protein complexes.
   The default generated YAMLs therefore omit the affinity property and can be
   used for cofolding, but they will not request affinity scalars or affinity-
   head embeddings.
2. **A custom protein-affinity runner is still required for the embedding
   experiment.** The generator exposes `--affinity-side` and
   `--binder-override` only for a runner known to support a protein binder.
   Stock Boltz will reject those YAMLs.
3. **Multi-chain binder semantics are unresolved.** `1AO7_ABC_DE` has
   multi-chain partners on both sides. Choosing D or E alone would represent
   only half of the T-cell receptor and is not scientifically neutral. A true
   group-binder mask is preferable if the custom affinity fork can support it.
4. **MSA policy must be fixed before the production run.** The YAMLs omit
   `msa`; stock execution therefore needs `--use_msa_server` or an explicit
   MSA/single-sequence policy. Re-querying MSAs for every mutant is expensive
   and can add variation unrelated to the mutation. Reusing a controlled WT
   MSA with an appropriately mutated query, or intentionally using
   single-sequence mode, should be evaluated.
5. **Detection-limit observations need a modeling policy.** The curated files
   contain weak-binding detection-cap notes and numeric capped values. The
   generator preserves them but does not infer censoring. Exact-value
   regression should either flag/exclude them in a sensitivity analysis or use
   a censored likelihood.
6. **The affinity head is out of domain even if made runnable.** Boltz-2's
   affinity module was trained for small-molecule binders. Protein-interface
   scalars and embeddings are exploratory and must be described as such.
7. **External preparation scripts are unavailable here.** The notes mention
   `run_babel.bat` and `runabcg.bat`, but neither is in this repository.
   Open-Babel SMILES/MOL2 conversion is probably irrelevant to these all-
   protein systems; the charge/force-field and ligand-residue interaction-
   energy role of `runabcg.bat` must be confirmed when Dad's workflow arrives.

## Planned per-system analysis

Build one model per system, analogous to the per-target Part-1 design, but with
continuous mutational effect as the primary task:

1. Collapse to one modeling row per unique `input_id`; use median
   delta-delta-G as the primary label and retain replicate spread/count.
2. Represent Boltz signal primarily as `embedding(mutant) - embedding(WT)`.
3. Compare a simple mutation-only baseline, embeddings alone, and their
   combination. Add LRIP/interaction-energy features when available.
4. Use strongly regularized Ridge or PLS models with nested CV because each
   system has only 47–275 unique mutants and the embeddings are high-
   dimensional.
5. Report out-of-fold Spearman as the headline, with delta-delta-G sign
   agreement, MAE, and Pearson as secondary metrics.
6. Never split repeated measurements of one `input_id` across folds. Random
   variant CV is the first analysis; position-held-out CV is a stricter
   secondary test for leakage through overlapping multi-mutants.
7. Use replicate disagreement to estimate assay noise and run median-vs-mean
   label sensitivity analyses before making claims about embedding gains.
