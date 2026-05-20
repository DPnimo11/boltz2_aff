"""Parse Keating-lab SORTCERY CSVs into a clean per-target mutational table.

Source data: papers/peptides/bh3/sortcery_design/csv/{x1,m1,f100}{,r}.csv
(Jenson et al. PNAS 2018; KeatingLab/sortcery_design GitHub).

Each input file contains apparent-affinity measurements of 22-mer BH3-family
peptides on yeast cell surface against one Bcl-2 family protein:
    x1 / x1r   -> Bcl-xL  (main + replicate sort, 1 nM target)
    x100/x100r -> Bcl-xL  (100 nM target)
    m1 / m1r   -> Mcl-1   (main + replicate)
    f100/f100r -> Bfl-1   (100 nM target; only concentration used in the paper)
    pilot_x1/* -> Bcl-xL  (early pilot screen, larger n, noisier)

Output: long-format TSV at data/peptides/bh3/measurements.tsv with one row
per (peptide, target, file) measurement. Columns:

    target            'Bcl-xL' | 'Mcl-1' | 'Bfl-1'
    source_file       basename of CSV (preserves replicate identity)
    is_replicate      bool; True for *r.csv files
    concentration_nM  1 | 100 (the displayed-target concentration)
    is_pilot          bool; True for pilot_x1*.csv
    peptide_seq       22-mer single-letter amino-acid string
    bg                'B' (Bim parent) | 'P' (PUMA parent)
    design_source     6-class label: <target>_<bg> ('m_b','m_p','x_b','x_p',
                                                     'f_b','f_p')
    apparent_value    SORTCERY apparent-affinity coordinate
                      (monotonic with log10 K_D on the cell surface; lower
                      = tighter on the sort axis - check sign convention
                      against the paper before interpreting absolute values)
    apparent_energy   apparent binding energy in arbitrary units (lower
                      = tighter)
    is_unimodal       quality flag from SORTCERY pipeline
    is_one_hit_wonder quality flag (True -> filter out)
    n_reads_total     total read count across sort bins (proxy for confidence)
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "papers" / "peptides" / "bh3" / "sortcery_design" / "csv"
OUT_DIR = REPO_ROOT / "data" / "peptides" / "bh3"


@dataclass(frozen=True)
class FileSpec:
    fname: str
    target: str
    prefix: str  # column-prefix used inside that CSV
    concentration_nM: int
    is_replicate: bool
    is_pilot: bool


SPECS: tuple[FileSpec, ...] = (
    FileSpec("x1.csv", "Bcl-xL", "x1", 1, False, False),
    FileSpec("x1r.csv", "Bcl-xL", "x1r", 1, True, False),
    FileSpec("x100.csv", "Bcl-xL", "x100", 100, False, False),
    FileSpec("x100r.csv", "Bcl-xL", "x100r", 100, True, False),
    FileSpec("pilot_x1.csv", "Bcl-xL", "pilot_x1", 1, False, True),
    FileSpec("pilot_x1r.csv", "Bcl-xL", "pilot_x1r", 1, True, True),
    FileSpec("m1.csv", "Mcl-1", "m1", 1, False, False),
    FileSpec("m1r.csv", "Mcl-1", "m1r", 1, True, False),
    FileSpec("f100.csv", "Bfl-1", "f100", 100, False, False),
    FileSpec("f100r.csv", "Bfl-1", "f100r", 100, True, False),
)

OUT_HEADER = (
    "target",
    "source_file",
    "is_replicate",
    "concentration_nM",
    "is_pilot",
    "peptide_seq",
    "bg",
    "design_source",
    "apparent_value",
    "apparent_energy",
    "is_unimodal",
    "is_one_hit_wonder",
    "n_reads_total",
)


def _parse_bool(s: str) -> bool:
    return s.strip().lower() == "true"


def _parse_file(spec: FileSpec) -> list[dict[str, str]]:
    path = SRC_DIR / spec.fname
    pref = spec.prefix
    out: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(
                {
                    "target": spec.target,
                    "source_file": spec.fname,
                    "is_replicate": str(spec.is_replicate),
                    "concentration_nM": str(spec.concentration_nM),
                    "is_pilot": str(spec.is_pilot),
                    "peptide_seq": row["protein"],
                    "bg": row["bg"],
                    "design_source": row["source"],
                    "apparent_value": row[f"{pref}_expectedValue"],
                    "apparent_energy": row[f"{pref}_energy"],
                    "is_unimodal": row[f"{pref}_isUnimodal"],
                    "is_one_hit_wonder": row[f"{pref}_isOneHitWonder"],
                    "n_reads_total": row[f"{pref}_CN_tot"],
                }
            )
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "measurements.tsv"

    all_rows: list[dict[str, str]] = []
    per_file: list[tuple[str, int]] = []
    for spec in SPECS:
        path = SRC_DIR / spec.fname
        if not path.exists():
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        rows = _parse_file(spec)
        per_file.append((spec.fname, len(rows)))
        all_rows.extend(rows)

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_HEADER, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"wrote {len(all_rows)} rows -> {out_path}")
    for fname, n in per_file:
        print(f"  {fname}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
