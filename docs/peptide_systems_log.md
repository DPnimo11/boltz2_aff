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
- Every generated protein chain explicitly uses `msa: empty`; this is an
  intentional single-sequence experiment and does not require an MSA server.
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

## Boltz fork audit and remaining execution issue

The local `../boltz` fork was inspected at commit `dacb835` ("update to save
affinity embeddings"). That patch is present and does exactly the needed
export work once the affinity path runs:

- `affinity.py` returns the pooled pre-MLP `g_pair_mean` and post-MLP `g_head`.
- `boltz2.py` propagates both representations for one- and two-model affinity
  execution.
- `writer.py` writes them to `affinity_embeddings_<record_id>.npz` beside the
  scalar affinity JSON.

The fact that the affinity head was trained on small molecules is **not an open
issue** here. Testing whether its representations remain informative for
protein/peptide mutation series is the purpose of the experiment. Results must
be interpreted as an out-of-domain stress test, but domain mismatch is not a
reason to stop.

Single-sequence execution is also resolved: every protein entry now has
`msa: empty`.

One code-level blocker remains in the checked-in fork: the export patch does
not extend affinity input semantics from a small-molecule ligand to a protein
partner.

1. `data/parse/schema.py` rejects an affinity binder unless it is one string
   naming an entity of type `ligand`.
2. The tokenizer creates `affinity_token_mask` for one `AffinityInfo.chain_id`.
3. The affinity module defines `rec_mask` as **all protein tokens** and assumes
   the affinity mask is a non-protein ligand. If a protein chain were simply
   marked as binder, binder tokens would overlap the receptor mask; the
   resulting cross-pair pooling would not cleanly mean binder-versus-partner.
4. The inference feature path always carries `affinity_mw`. Molecular-weight
   correction is off by default, but a protein-binder patch still needs a
   defined numeric/optional value that survives batching.
5. `1AO7_ABC_DE` requires a group binder because both partners are multi-chain.
   Selecting D or E alone represents only half of the T-cell receptor. The
   schema, `AffinityInfo`, tokenizer mask, and pooling path currently support
   only one chain.

Therefore the generated YAMLs still omit `properties.affinity` by default and
can run the structure path in single-sequence mode, but they cannot trigger the
affinity embedding writer with the checked-in fork alone. If another
protein-binder patch already exists in the production environment, it should
be synchronized or documented here. Otherwise the next implementation step is
to add a binder-chain-group mask, define `rec_mask = protein_mask &
~binder_mask`, and make molecular weight optional for embedding-only PPI runs.

There is evidence that an additional peptide extraction path existed: the
2,139 prior BH3/p53 artifacts contain `pair_mean`, `head_ens1`, `head_ens2`,
`head_mean`, `peptide_id`, and `target`, whereas this fork's writer emits raw
`affinity_embedding_pair_mean{1,2}` / `affinity_embedding_head{1,2}` keys. No
producer for that normalized peptide schema is present in either checked tree
or in any local/remote `../boltz` branch. Locating that script or patch is the
main remaining provenance question; it may already implement the missing
protein-binder behavior.

## Remaining analysis issue

**Detection-limit observations need a modeling policy.** The curated files
contain weak-binding detection-cap notes and numeric capped values. The
generator preserves them but does not infer censoring. Exact-value regression
should either flag/exclude them in a sensitivity analysis or use a censored
likelihood. Repeated measurements are already preserved and summarized; their
primary median-label and sensitivity-analysis policy is described above.

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
