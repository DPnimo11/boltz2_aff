"""Transcribe Tables 1 and 3 from Li, Pazgier et al. J Mol Biol 2010
(PMC2856455, PMID 20226197) into a clean TSV at
``data/peptides/source/p53/measurements.tsv``.

The source tables are rendered as images inside the PDF and cannot be
extracted with pdf-text tools - rendered crops are checked into
``data/peptides/source/p53/raw/`` for reference. The numeric values here were
transcribed once from those rendered tables.

Two peptide scaffolds, two receptors each:

  * PMI scaffold        = TSFAEYWNLLSP   (Table 1; 12 Ala-scan analogs +
                                          5 truncation analogs)
  * (17-28)p53 scaffold = ETFSDLWKLLPE   (Table 3; 12 Ala-scan analogs +
                                          5 truncation analogs)

Receptors:
  * synMDM2  (residues 25-109 of MDM2, chemically synthesized)
  * synMDMX  (residues 24-108 of MDMX, chemically synthesized)

Affinity assay: SPR-based competition binding (Methods, page 12 of the PDF).
At least three independent measurements per peptide; mean +/- SD reported.

The Ala4 substitution of PMI (position 4 is already Ala in the wild-type)
is a no-op and is included in the source table as a consistency control;
we tag it ``analog_class='control_redundant'`` so downstream code can
exclude it from mutational-effect analyses.

F19A and W23A of (17-28)p53 bound MDM2/MDMX too weakly to quantify
(>3 orders of magnitude weaker than parent); we encode these as
``kd_M = None`` with ``analog_class='not_determined'``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "peptides" / "source" / "p53"


@dataclass(frozen=True)
class Row:
    scaffold: str               # 'PMI' or 'p53_17_28'
    scaffold_parent_seq: str
    analog_class: str           # 'parent' | 'ala_scan' | 'truncation'
                                # | 'control_redundant' | 'not_determined'
    mutation_label: str         # 'WT', 'T1A', 'S2A', '1-10', '17-26', ...
    peptide_seq: str
    receptor: str               # 'MDM2' or 'MDMX'
    kd_M: float | None
    kd_sd_M: float | None
    kd_ratio_vs_parent: float | None
    ddG_kcal_per_mol: float | None


# -----------------------------------------------------------------------
# Table 1 (PMI scaffold) -- transcribed from the page-2 image.
# Each entry: (analog_class, label, sequence,
#              (kd_MDM2_M, sd_MDM2_M, ratio_MDM2, ddG_MDM2),
#              (kd_MDMX_M, sd_MDMX_M, ratio_MDMX, ddG_MDMX))
# Kd values written in scientific notation (mantissa * 10**exp).
# -----------------------------------------------------------------------

PMI = "TSFAEYWNLLSP"

PMI_TABLE = [
    ("parent", "WT", "TSFAEYWNLLSP",
     (3.2e-9, 1.1e-9, 1.0, 0.00),
     (8.5e-9, 1.7e-9, 1.0, 0.00)),
    ("ala_scan", "T1A", "ASFAEYWNLLSP",
     (6.2e-9, 0.1e-9, 1.9, 0.39),
     (1.6e-8, 0.3e-8, 1.8, 0.35)),
    ("ala_scan", "S2A", "TAFAEYWNLLSP",
     (2.7e-8, 0.4e-8, 8.4, 1.24),
     (3.7e-8, 0.4e-8, 4.3, 0.85)),
    ("ala_scan", "F3A", "TSAAEYWNLLSP",
     (3.8e-5, 0.2e-5, 11750.0, 5.46),
     (1.2e-4, 0.1e-4, 14120.0, 5.57)),
    # Position 4 of PMI is already Ala -- table includes it as a
    # consistency control; same sequence/values as WT.
    ("control_redundant", "A4A", "TSFAEYWNLLSP",
     (3.2e-9, 1.1e-9, 1.0, 0.00),
     (8.5e-9, 1.7e-9, 1.0, 0.00)),
    ("ala_scan", "E5A", "TSFAAYWNLLSP",
     (2.1e-8, 0.1e-8, 6.7, 1.10),
     (6.7e-8, 0.9e-8, 7.8, 1.20)),
    ("ala_scan", "Y6A", "TSFAEAWNLLSP",
     (6.1e-7, 0.7e-7, 191.0, 3.06),
     (6.7e-7, 0.8e-7, 79.0, 2.55)),
    ("ala_scan", "W7A", "TSFAEYANLLSP",
     (1.6e-4, 0.3e-4, 50720.0, 6.31),
     (2.3e-4, 0.1e-4, 26590.0, 5.94)),
    ("ala_scan", "N8A", "TSFAEYWALLSP",
     (4.9e-10, 2.1e-10, 0.2, -1.10),
     (2.4e-9, 0.6e-9, 0.3, -0.74)),
    ("ala_scan", "L9A", "TSFAEYWNALSP",
     (2.4e-9, 0.5e-9, 0.8, -0.17),
     (9.0e-9, 2.1e-9, 1.1, 0.03)),
    ("ala_scan", "L10A", "TSFAEYWNLASP",
     (8.9e-7, 0.1e-7, 277.0, 3.28),
     (4.3e-7, 0.4e-7, 50.0, 2.28)),
    ("ala_scan", "S11A", "TSFAEYWNLLAP",
     (3.9e-9, 0.3e-9, 1.2, 0.12),
     (1.1e-8, 0.2e-8, 1.3, 0.17)),
    ("ala_scan", "P12A", "TSFAEYWNLLSA",
     (2.1e-9, 0.5e-9, 0.7, -0.25),
     (1.4e-8, 0.3e-8, 1.7, 0.31)),
    # Truncation analogs.
    ("truncation", "1-10", "TSFAEYWNLL",
     (8.6e-9, 0.6e-9, 2.7, 0.58),
     (2.9e-8, 0.5e-8, 3.4, 0.71)),
    ("truncation", "2-10", "SFAEYWNLL",
     (1.7e-7, 0.1e-7, 53.0, 2.31),
     (6.7e-7, 0.6e-7, 79.0, 2.55)),
    ("truncation", "3-10", "FAEYWNLL",
     (8.9e-6, 0.7e-6, 2780.0, 4.62),
     (4.4e-5, 0.5e-5, 5180.0, 4.98)),
    ("truncation", "3-11", "FAEYWNLLS",
     (1.4e-5, 0.2e-5, 4375.0, 4.88),
     (3.7e-5, 0.1e-5, 4350.0, 4.88)),
    ("truncation", "3-12", "FAEYWNLLSP",
     (6.5e-6, 1.1e-6, 2030.0, 4.44),
     (8.8e-6, 1.0e-6, 1035.0, 4.04)),
]


# -----------------------------------------------------------------------
# Table 3 ((17-28)p53 scaffold) -- transcribed from the page-7 image.
# Position labels use p53 residue numbering (17-28); local positions
# are 1-12 within the 12-mer.
# -----------------------------------------------------------------------

P53_17_28 = "ETFSDLWKLLPE"

P53_TABLE = [
    ("parent", "WT", "ETFSDLWKLLPE",
     (4.4e-7, 0.4e-7, 1.0, 0.00),
     (6.4e-7, 0.5e-7, 1.0, 0.00)),
    ("ala_scan", "E17A", "ATFSDLWKLLPE",
     (5.6e-7, 0.2e-7, 1.3, 0.14),
     (6.8e-7, 0.1e-7, 1.1, 0.03)),
    ("ala_scan", "T18A", "EAFSDLWKLLPE",
     (1.2e-6, 0.1e-6, 2.7, 0.58),
     (2.3e-6, 0.1e-6, 3.6, 0.75)),
    # F19A: too weak to quantify by SPR (>3 orders of magnitude weaker).
    ("not_determined", "F19A", "ETASDLWKLLPE",
     (None, None, None, None),
     (None, None, None, None)),
    ("ala_scan", "S20A", "ETFADLWKLLPE",
     (2.1e-7, 0.1e-7, 0.5, -0.43),
     (3.1e-7, 0.1e-7, 0.5, -0.43)),
    ("ala_scan", "D21A", "ETFSALWKLLPE",
     (8.3e-7, 0.2e-7, 1.9, 0.37),
     (1.1e-6, 0.1e-6, 1.7, 0.32)),
    ("ala_scan", "L22A", "ETFSDAWKLLPE",
     (5.0e-6, 0.4e-6, 11.0, 1.41),
     (9.0e-6, 0.8e-6, 14.0, 1.54)),
    # W23A: too weak to quantify by SPR.
    ("not_determined", "W23A", "ETFSDLAKLLPE",
     (None, None, None, None),
     (None, None, None, None)),
    ("ala_scan", "K24A", "ETFSDLWALLPE",
     (2.3e-7, 0.2e-7, 0.5, -0.39),
     (4.9e-7, 0.4e-7, 0.8, -0.15)),
    ("ala_scan", "L25A", "ETFSDLWKALPE",
     (7.3e-7, 0.1e-7, 1.7, 0.30),
     (6.9e-7, 0.6e-7, 1.1, 0.04)),
    ("ala_scan", "L26A", "ETFSDLWKLAPE",
     (2.7e-5, 0.1e-5, 61.0, 2.39),
     (6.6e-5, 0.1e-5, 102.0, 2.70)),
    ("ala_scan", "P27A", "ETFSDLWKLLAE",
     (5.1e-8, 0.3e-8, 0.1, -1.26),
     (2.4e-7, 0.2e-7, 0.4, -0.58)),
    ("ala_scan", "E28A", "ETFSDLWKLLPA",
     (2.4e-7, 0.2e-7, 0.5, -0.36),
     (3.3e-7, 0.1e-7, 0.5, -0.39)),
    # Truncation analogs.
    ("truncation", "17-26", "ETFSDLWKLL",
     (7.5e-8, 0.2e-8, 0.2, -1.03),
     (3.9e-7, 0.2e-7, 0.6, -0.29)),
    ("truncation", "18-26", "TFSDLWKLL",
     (1.0e-6, 0.1e-6, 2.3, 0.47),
     (2.4e-6, 0.1e-6, 3.8, 0.77)),
    ("truncation", "19-26", "FSDLWKLL",
     (3.5e-5, 0.4e-5, 79.0, 2.55),
     (1.3e-4, 0.2e-4, 195.0, 3.07)),
    ("truncation", "19-27", "FSDLWKLLP",
     (1.4e-4, 0.3e-4, 319.0, 3.36),
     (1.6e-4, 0.1e-4, 244.0, 3.20)),
    ("truncation", "19-28", "FSDLWKLLPE",
     (1.2e-4, 0.2e-4, 269.0, 3.26),
     (1.9e-4, 0.4e-4, 291.0, 3.30)),
]


def _expand(scaffold: str, parent: str, table) -> list[Row]:
    out: list[Row] = []
    for analog_class, label, seq, mdm2, mdmx in table:
        kd1, sd1, ratio1, ddG1 = mdm2
        kd2, sd2, ratio2, ddG2 = mdmx
        out.append(Row(scaffold, parent, analog_class, label, seq,
                       "MDM2", kd1, sd1, ratio1, ddG1))
        out.append(Row(scaffold, parent, analog_class, label, seq,
                       "MDMX", kd2, sd2, ratio2, ddG2))
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    rows.extend(_expand("PMI", PMI, PMI_TABLE))
    rows.extend(_expand("p53_17_28", P53_17_28, P53_TABLE))

    out_path = OUT_DIR / "measurements.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "scaffold", "scaffold_parent_seq", "analog_class",
            "mutation_label", "peptide_seq", "receptor",
            "kd_M", "kd_sd_M", "kd_ratio_vs_parent", "ddG_kcal_per_mol",
        ])
        for r in rows:
            writer.writerow([
                r.scaffold, r.scaffold_parent_seq, r.analog_class,
                r.mutation_label, r.peptide_seq, r.receptor,
                "" if r.kd_M is None else f"{r.kd_M:.3e}",
                "" if r.kd_sd_M is None else f"{r.kd_sd_M:.3e}",
                "" if r.kd_ratio_vs_parent is None else f"{r.kd_ratio_vs_parent:g}",
                "" if r.ddG_kcal_per_mol is None else f"{r.ddG_kcal_per_mol:.2f}",
            ])

    n_total = len(rows)
    n_nd = sum(1 for r in rows if r.kd_M is None)
    print(f"wrote {n_total} rows -> {out_path}")
    print(f"  scaffolds: PMI ({len(PMI_TABLE)} entries), "
          f"p53_17_28 ({len(P53_TABLE)} entries); 2 receptors each")
    print(f"  rows with measurable Kd: {n_total - n_nd}")
    print(f"  rows tagged not_determined: {n_nd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
