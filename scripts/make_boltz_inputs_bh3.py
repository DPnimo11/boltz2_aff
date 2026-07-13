"""Generate Boltz-2 cofolding inputs for the BH3 peptide system.

Scope (chosen 2026-05-20): the *cross-target* subset - the set of
22-mer peptides that appear in ALL THREE non-pilot SORTCERY datasets
(x1.csv for Bcl-xL, m1.csv for Mcl-1, f100.csv for Bfl-1). This yields
the cleanest comparable selectivity slice with the smallest cofolding
workload. Replicate (*r.csv) and pilot files are pooled here only to
enrich the intersection check; the YAMLs themselves are deduplicated by
peptide sequence and only cover the cross-target peptides.

Output layout (one folder per receptor):

    data/peptides/boltz/inputs/bh3/<receptor>/input/<peptide_id>.yaml
    data/peptides/boltz/inputs/bh3/<receptor>/manifest.tsv

Plus a global ``data/peptides/boltz/inputs/bh3/peptide_index.tsv`` that maps
the canonical peptide_id <-> sequence so the same ID is used across
receptors.

Receptor binding-domain constructs (matching the published purification
constructs used in the Keating lab SORTCERY measurements):

    Bcl-xL -> residues 1-209 (DeltaTM)        UniProt Q07817
    Mcl-1  -> residues 172-327 (DeltaN DeltaC)  UniProt Q07820
    Bfl-1  -> residues 1-151 (DeltaTM)        UniProt Q16548
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "peptides" / "source" / "bh3" / "measurements.tsv"
OUT_ROOT = REPO_ROOT / "data" / "peptides" / "boltz" / "inputs" / "bh3"


# UniProt SV=1 / SV=3 full-length sequences; slices below are the binding-
# domain constructs used in the published Keating-lab SORTCERY measurements.
_BCLXL_FULL = (
    "MSQSNRELVVDFLSYKLSQKGYSWSQFSDVEENRTEAPEGTESEMETPSAINGNPSWHLA"
    "DSPAVNGATGHSSSLDAREVIPMAAVKQALREAGDEFELRYRRAFSDLTSQLHITPGTAY"
    "QSFEQVVNELFRDGVNWGRIVAFFSFGGALCVESVDKEMQVLVSRIAAWMATYLNDHLEP"
    "WIQENGGWDTFVELYGNNAAAESRKGQERFNRWFLTGMTVAGVVLLGSLFSRK"
)
_MCL1_FULL = (
    "MFGLKRNAVIGLNLYCGGAGLGAGSGGATRPGGRLLATEKEASARREIGGGEAGAVIGGS"
    "AGASPPSTLTPDSRRVARPPPIGAEVPDVTATPARLLFFAPTRRAAPLEEMEAPAADAIM"
    "SPEEELDGYEPEPLGKRPAVLPLLELVGESGNNTSTDGSLPSTPPPAEEEEDELYRQSLE"
    "IISRYLREQATGAKDTKPMGRSGATSRKALETLRRVGDGVQRNHETAFQGMLRKLDIKNE"
    "DDVKSLSRVMIHVFSDGVTNWGRIVTLISFGAFVAKHLKTINQESCIEPLAESITDVLVR"
    "TKRDWLVKQRGWDGFVEFFHVEDLEGGIRNVLLAFAGVAGVGAGLAYLIR"
)
_BFL1_FULL = (
    "MTDCEFGYIYRLAQDYLQCVLQIPQPGSGPSKTSRVLQNVAFSVQKEVEKNLKSCLDNVN"
    "VVSVDTARTLFNQVMEKEFEDGIINWGRIVTIFAFEGILIKKLLRQQIAPDVDTYKEISY"
    "FVAEFIMNNTGEWIRQNGGWENGFVKKFEPKSGWMTFLEVTGKICEMLSLLKQYC"
)

RECEPTOR_SEQS: dict[str, dict[str, object]] = {
    "Bcl-xL": {
        "uniprot": "Q07817",
        "range": (1, 209),
        "sequence": _BCLXL_FULL[0:209],
    },
    "Mcl-1": {
        "uniprot": "Q07820",
        "range": (172, 327),
        "sequence": _MCL1_FULL[171:327],
    },
    "Bfl-1": {
        "uniprot": "Q16548",
        "range": (1, 151),
        "sequence": _BFL1_FULL[0:151],
    },
}


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


def _load_unique_peptides_per_target() -> dict[str, set[str]]:
    """Return {target: set(peptide_seq)} from the primary SORTCERY sorts.

    Restricted to ``is_pilot=False`` AND ``is_replicate=False`` - i.e.,
    the canonical x1.csv / m1.csv / f100.csv main sorts (the
    1 nM-displayed-target main run for Bcl-xL and Mcl-1; the 100 nM main
    run for Bfl-1). Replicate-sort and pilot files are intentionally
    *not* used for the intersection: replicates cover different
    stochastic samples of the library and would inflate the cross-target
    set well beyond the headline ~1.6k. Replicate measurements remain
    available in ``data/peptides/source/bh3/measurements.tsv`` for noise
    estimation downstream.
    """
    out: dict[str, set[str]] = {"Bcl-xL": set(), "Mcl-1": set(), "Bfl-1": set()}
    with SRC.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row["is_pilot"] == "True" or row["is_replicate"] == "True":
                continue
            target = row["target"]
            if target in out:
                out[target].add(row["peptide_seq"])
    return out


def _provenance_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    """Compact provenance: which source files and bg/design_source codes
    contained this peptide measurement (across all targets)."""
    src_files = sorted({r["source_file"] for r in rows})
    bgs = sorted({r["bg"] for r in rows})
    sources = sorted({r["design_source"] for r in rows})
    return {
        "source_files": ",".join(src_files),
        "bg": ",".join(bgs),
        "design_source": ",".join(sources),
    }


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect all measurement rows once, grouped by peptide_seq.
    rows_by_peptide: dict[str, list[dict[str, str]]] = {}
    with SRC.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows_by_peptide.setdefault(row["peptide_seq"], []).append(row)

    unique_per_target = _load_unique_peptides_per_target()
    cross_target = (unique_per_target["Bcl-xL"]
                    & unique_per_target["Mcl-1"]
                    & unique_per_target["Bfl-1"])
    print(f"cross-target peptides (in Bcl-xL & Mcl-1 & Bfl-1): {len(cross_target)}")

    # Global ID assignment: alphabetical ordering of the peptide sequences
    # gives a stable, reproducible peptide_id across runs.
    ordered = sorted(cross_target)
    peptide_id_of = {seq: f"bh3_{i:04d}" for i, seq in enumerate(ordered)}

    # Write the global peptide index.
    index_path = OUT_ROOT / "peptide_index.tsv"
    with index_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["peptide_id", "peptide_seq", "source_files", "bg",
                         "design_source"])
        for seq in ordered:
            prov = _provenance_summary(rows_by_peptide[seq])
            writer.writerow([
                peptide_id_of[seq], seq,
                prov["source_files"], prov["bg"], prov["design_source"],
            ])
    print(f"wrote {index_path}")

    # Per-receptor: YAMLs + manifest of (peptide_id, sequence,
    # apparent_value, apparent_energy) for that target only.
    for receptor, info in RECEPTOR_SEQS.items():
        receptor_seq = info["sequence"]
        assert isinstance(receptor_seq, str)
        input_dir = OUT_ROOT / receptor / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, str]] = []
        for seq in ordered:
            pid = peptide_id_of[seq]
            yaml_path = input_dir / f"{pid}.yaml"
            yaml_path.write_text(_yaml(receptor_seq, seq), encoding="utf-8")

            # Pull the best (non-pilot, non-replicate) measurement for the
            # manifest. Fall back to non-pilot replicate, then anything.
            measurements = [r for r in rows_by_peptide[seq]
                            if r["target"] == receptor]
            measurements.sort(key=lambda r: (r["is_pilot"], r["is_replicate"]))
            best = measurements[0] if measurements else None
            manifest_rows.append({
                "peptide_id": pid,
                "peptide_seq": seq,
                "apparent_value": best["apparent_value"] if best else "",
                "apparent_energy": best["apparent_energy"] if best else "",
                "source_file": best["source_file"] if best else "",
                "bg": best["bg"] if best else "",
                "design_source": best["design_source"] if best else "",
                "n_reads_total": best["n_reads_total"] if best else "",
            })

        manifest_path = OUT_ROOT / receptor / "manifest.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh,
                                    fieldnames=list(manifest_rows[0].keys()),
                                    delimiter="\t")
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"{receptor}: {len(ordered)} YAMLs + {manifest_path}")

    print(f"\nTotal cofolds queued: {len(ordered) * len(RECEPTOR_SEQS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
