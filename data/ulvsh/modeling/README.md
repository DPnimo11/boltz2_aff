# ULVSH modeling data

This directory contains the compact, versioned inputs used by the Part 1
models. The imported ULVSH dataset remains under `../source/`, while the
transferred paper/reference Boltz inputs remain under `../reference_boltz/`.

## Files

| Path | Role |
|---|---|
| `labels.tsv` | One normalized experimental-label row per `(target, ligand_id)`, derived from the ULVSH `raw/vitro.tsv` files. |
| `features/boltz_embeddings/<target>/` | Affinity embeddings produced with the modified exporter and consumed by the embedding models. |
| `features/boltz_scalars.tsv` | All six numeric affinity fields from the paper/reference run, one row per `(target, variant, ligand_id)`. |
| `manifest.tsv` | Per-target coverage for labels, embeddings, scalar variants, and retained reference-run YAMLs. |

`boltz_scalars.tsv` exactly round-trips the 2,830 transferred affinity JSON
records. It is not a fitted model or a new prediction. During consolidation,
74 CASR shuffled records found under the transferred misspelling
`shuffled/ouput/` were correctly labeled `variant=shuffled`; the legacy path
loader had previously inferred those rows as WT.

The large paper/reference output tree is intentionally not retained. PAE, PDE,
predicted PDB, pLDDT, confidence, and `pre_affinity` files are not used by the
current models. The compact scalar table is sufficient for the raw B2-A/B2-C
baseline and combined feature sets.
