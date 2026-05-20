"""Generate Boltz-2 cofolding input YAMLs for the p53 peptide system.

For every (peptide_seq, receptor) pair that appears in
``data/peptides/p53/measurements.tsv`` we emit one YAML at:

    data/Boltz-2/peptides/p53/<receptor>/input/<peptide_id>.yaml

Receptor names map to the *synthetic* binding-domain constructs used by
Pazgier 2009 / Li 2010 (the source of the Kd values):

    MDM2  -> synMDM2  = MDM2 residues 25-109   (UniProt Q00987)
    MDMX  -> synMDMX  = MDMX residues 24-108   (UniProt O15151)

Peptide identifiers concatenate the scaffold and the mutation label, with
``/`` and ``-`` made filesystem-safe. The PMI A4A row is a no-op control
(position 4 is already Ala); the F19A/W23A rows on the p53 scaffold are
``not_determined`` in the source - all of these still get a YAML so the
Boltz prediction is recorded, but they are tagged in the manifest.

Outputs a per-receptor ``manifest.tsv`` next to the input directory mapping
peptide_id <-> peptide_seq + provenance.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "peptides" / "p53" / "measurements.tsv"
OUT_ROOT = REPO_ROOT / "data" / "Boltz-2" / "peptides" / "p53"


# Synthetic MDM2 / MDMX binding-domain constructs used by Pazgier 2009 and
# Li 2010 (the source papers for the Kd values we're trying to predict).
# Residue ranges below are 1-indexed inclusive against the canonical
# UniProt entries.
RECEPTOR_SEQS: dict[str, dict[str, str | tuple[int, int]]] = {
    "MDM2": {
        "uniprot": "Q00987",
        "range": (25, 109),
        # Q00987 residues 25-109 inclusive (85 aa); verified against UniProt SV=1.
        "sequence": (
            "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEK"
            "QQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV"
        ),
    },
    "MDMX": {
        "uniprot": "O15151",
        "range": (24, 108),
        # O15151 residues 24-108 inclusive (85 aa); verified against UniProt SV=2.
        "sequence": (
            "INQVRPKLPLLKILHAAGAQGEMFTVKEVMHYLGQYIMVKQLYDQQ"
            "EQHMVYCGGDLLGELLGRQSFSVKDPSPLYDMLRKNLVT"
        ),
    },
}


def _peptide_id(scaffold: str, mutation_label: str) -> str:
    safe = mutation_label.replace("-", "_").replace("/", "_")
    return f"{scaffold}_{safe}"


def _yaml(receptor_seq: str, peptide_seq: str) -> str:
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {receptor_seq}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {peptide_seq}\n"
        "properties:\n"
        "  - affinity:\n"
        "      binder: B\n"
    )


def main() -> int:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8"), delimiter="\t"))

    # Collapse to unique (scaffold, mutation_label) entries per receptor;
    # the source TSV has one row per (peptide, receptor) which is exactly
    # what we want, but PMI/A4A has the same sequence as PMI/WT so we
    # still write a separate YAML (different ID, identical content) to
    # match the Li 2010 row inventory.
    written: dict[str, int] = {r: 0 for r in RECEPTOR_SEQS}
    manifests: dict[str, list[dict[str, str]]] = {
        r: [] for r in RECEPTOR_SEQS
    }

    for row in rows:
        receptor = row["receptor"]
        if receptor not in RECEPTOR_SEQS:
            raise ValueError(f"unknown receptor {receptor!r}")
        receptor_seq = RECEPTOR_SEQS[receptor]["sequence"]
        peptide_id = _peptide_id(row["scaffold"], row["mutation_label"])
        peptide_seq = row["peptide_seq"]

        input_dir = OUT_ROOT / receptor / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = input_dir / f"{peptide_id}.yaml"
        yaml_path.write_text(_yaml(receptor_seq, peptide_seq), encoding="utf-8")
        written[receptor] += 1

        manifests[receptor].append({
            "peptide_id": peptide_id,
            "peptide_seq": peptide_seq,
            "scaffold": row["scaffold"],
            "scaffold_parent_seq": row["scaffold_parent_seq"],
            "mutation_label": row["mutation_label"],
            "analog_class": row["analog_class"],
            "kd_M": row["kd_M"],
            "kd_sd_M": row["kd_sd_M"],
            "ddG_kcal_per_mol": row["ddG_kcal_per_mol"],
        })

    for receptor, entries in manifests.items():
        mpath = OUT_ROOT / receptor / "manifest.tsv"
        with mpath.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(entries[0].keys()),
                                    delimiter="\t")
            writer.writeheader()
            writer.writerows(entries)
        print(f"{receptor}: {written[receptor]} YAMLs + {mpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
