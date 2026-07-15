# LRIP interaction-profile features

Ligand–residue interaction profiles (LRIP / IP-SF, Junmei Wang lab) for the 10
ULVSH targets. One `<TARGET>.dat` per target, transferred from the pose +
MM-GBSA decomposition run (originally landed in `_sftp_lrip/`).

## File format

Whitespace-delimited. The first row is a header: the literal token `Comp_ID`
followed by receptor residue numbers (the per-residue interaction-energy
columns). Residue numbers may repeat (e.g. `81 81`) — these are distinct energy
columns and are kept as separate features. Each subsequent row is one compound:
its ligand id followed by the per-residue interaction energies (kcal/mol; more
negative = more favorable). Residue sets differ per target, so LRIP is a
per-target feature block.

| Target | compounds | residue columns |
|---|---|---|
| ADRA2B | 13 | 48 |
| CASR | 74 | 54 |
| CNR1 | 45 | 64 |
| CNR2 | 58 | 78 |
| DRD3 | 33 | 56 |
| DRD4 | 324 | 75 |
| MTR1A | 36 | 55 |
| ROCK1 | 68 | 85 |
| SC6A4 | 32 | 65 |
| SGMR2 | 200 | 60 |

The compound set per target is the Boltz WT-input subset (`n_input_wt` in
`../../manifest.tsv`); a handful of compounds are absent because their LRIP /
MM-GBSA decomposition did not complete.

## Joining to labels

Row ids join directly to `../../labels.tsv` on `(target, ligand_id)` — including
ROCK1, whose canonical ULVSH ligand ids are themselves `mol_01…mol_69`. Two
things to know when joining:

- **DRD3:** the `.dat` writes one id as `1_20` where `labels.tsv` has `1_2_0`
  (a dropped underscore; one-to-one, same compound). The modeling script
  normalizes `1_20 -> 1_2_0` for DRD3.
- **ROCK1:** `mol_44` is absent from the `.dat` (LRIP failure); the other 68 of
  69 join cleanly.

With those handled, every LRIP row maps to a label (0 unmatched across all
targets).

Consumed by `scripts/model_lrip.py`.
