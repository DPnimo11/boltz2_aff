"""Generate mutation-resolved Boltz cofolding inputs for the SKEMPI subset.

The source bundles live under ``data/peptide_systems/systems/<PDB>/``.  Each
bundle provides chain sequences in ``<PDB>.fasta`` and experimental variants
in one curated ``<PDB>_<group1>_<group2>_New.txt`` table.

Mutation numbers match the fourth ``.mapping`` field (``SEQIDX``), not the
third field (PDB residue number).  In 12 systems the mapping sequence and FASTA
are identical, so ``SEQIDX`` is also the FASTA index.  3S9D contains additional
FASTA residues; its mapping sequence is aligned to the FASTA before mutation
coordinates are applied.

Repeated experiments are deduplicated only for structure generation.  One
YAML is written per unique canonical mutation set (plus WT), while
``measurements.tsv`` retains every curated experimental row and maps it to the
shared ``input_id``.  ``variants.tsv`` contains one row per generated YAML and
summarizes replicate delta-delta-G values without discarding the originals.

By default the YAMLs request structure prediction only.  Stock Boltz-2 accepts
only a single small-molecule affinity binder, whereas these are protein-protein
systems and some partners span multiple chains.  ``--affinity-side`` is
therefore an explicit custom-runner option, not the default.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEMS_ROOT = REPO_ROOT / "data" / "peptide_systems" / "systems"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "peptide_systems" / "boltz_inputs"

AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
MUTATION_RE = re.compile(
    r"^(?P<wt>[ACDEFGHIKLMNPQRSTVWY])"
    r"(?P<chain>[A-Za-z0-9])"
    r"(?P<position>[1-9][0-9]*)"
    r"(?P<mut>[ACDEFGHIKLMNPQRSTVWY])$"
)
DATA_ID_RE = re.compile(r"^[A-Za-z0-9]{4}_[0-9]+$")


@dataclass(frozen=True, order=True)
class Mutation:
    """One amino-acid substitution against a one-based FASTA position."""

    chain: str
    position: int
    wt: str
    mut: str

    @property
    def text(self) -> str:
        return f"{self.wt}{self.chain}{self.position}{self.mut}"


@dataclass(frozen=True)
class SystemSpec:
    pdb_id: str
    name: str
    source_dir: Path
    curated_path: Path
    fasta_path: Path
    group1: tuple[str, ...]
    group2: tuple[str, ...]
    sequences: dict[str, str]
    seqidx_to_fasta_position: dict[tuple[str, int], int]

    @property
    def included_chains(self) -> tuple[str, ...]:
        wanted = set(self.group1 + self.group2)
        return tuple(chain for chain in self.sequences if chain in wanted)

    @property
    def excluded_chains(self) -> tuple[str, ...]:
        wanted = set(self.group1 + self.group2)
        return tuple(chain for chain in self.sequences if chain not in wanted)

    @property
    def mapping_adjusted_chains(self) -> tuple[str, ...]:
        return tuple(
            chain
            for chain in self.included_chains
            if any(
                seqidx != fasta_position
                for (mapped_chain, seqidx), fasta_position
                in self.seqidx_to_fasta_position.items()
                if mapped_chain == chain
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems-root",
        type=Path,
        default=DEFAULT_SYSTEMS_ROOT,
        help="source system-bundle root",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="generated YAML/manifest root",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        help="optional PDB ids or full system names to generate",
    )
    parser.add_argument(
        "--affinity-side",
        choices=("none", "group1", "group2"),
        default="none",
        help=(
            "custom-runner only: add properties.affinity using the selected "
            "partner side; that side must contain exactly one chain"
        ),
    )
    parser.add_argument(
        "--binder-override",
        action="append",
        default=[],
        metavar="SYSTEM=CHAIN",
        help=(
            "custom-runner only: override the affinity binder for one system; "
            "repeat as needed"
        ),
    )
    return parser.parse_args()


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, list[str]] = {}
    chain: str | None = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if ":" not in line:
                raise ValueError(f"{path}:{line_number}: FASTA header lacks chain id")
            chain = line.rsplit(":", 1)[1].strip()
            if len(chain) != 1:
                raise ValueError(
                    f"{path}:{line_number}: expected one-character chain id, got {chain!r}"
                )
            if chain in sequences:
                raise ValueError(f"{path}:{line_number}: duplicate chain {chain!r}")
            sequences[chain] = []
            continue
        if chain is None:
            raise ValueError(f"{path}:{line_number}: sequence precedes first header")
        sequence_line = line.upper()
        invalid = set(sequence_line) - AA
        if invalid:
            raise ValueError(
                f"{path}:{line_number}: unsupported residues {sorted(invalid)}"
            )
        sequences[chain].append(sequence_line)

    joined = {name: "".join(parts) for name, parts in sequences.items()}
    if not joined or any(not sequence for sequence in joined.values()):
        raise ValueError(f"{path}: missing FASTA chain or sequence")
    return joined


def read_mapping_positions(
    path: Path, sequences: dict[str, str]
) -> dict[tuple[str, int], int]:
    """Map each mapping SEQIDX to its aligned one-based FASTA position."""

    mapped_residues: dict[str, dict[int, str]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: expected four mapping fields")
        residue_name, chain, _pdb_number, raw_seqidx = fields
        if residue_name not in THREE_TO_ONE:
            raise ValueError(
                f"{path}:{line_number}: unsupported residue {residue_name!r}"
            )
        try:
            seqidx = int(raw_seqidx)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid SEQIDX {raw_seqidx!r}"
            ) from exc
        chain_map = mapped_residues.setdefault(chain, {})
        if seqidx in chain_map:
            raise ValueError(f"{path}:{line_number}: duplicate {chain}:{seqidx}")
        chain_map[seqidx] = THREE_TO_ONE[residue_name]

    coordinate_map: dict[tuple[str, int], int] = {}
    for chain, by_index in mapped_residues.items():
        if chain not in sequences:
            raise ValueError(f"{path}: mapping chain {chain!r} is absent from FASTA")
        expected_indices = list(range(1, len(by_index) + 1))
        if sorted(by_index) != expected_indices:
            raise ValueError(f"{path}: {chain} SEQIDX values are not contiguous from 1")
        mapped_sequence = "".join(by_index[index] for index in expected_indices)
        fasta_sequence = sequences[chain]

        matcher = difflib.SequenceMatcher(
            None, mapped_sequence, fasta_sequence, autojunk=False
        )
        for tag, map_start, map_end, fasta_start, fasta_end in matcher.get_opcodes():
            if tag == "insert":
                continue
            if tag != "equal":
                raise ValueError(
                    f"{path}: mapping chain {chain} is not an exact subsequence of "
                    f"the FASTA (alignment operation {tag!r})"
                )
            for offset in range(map_end - map_start):
                seqidx = map_start + offset + 1
                fasta_position = fasta_start + offset + 1
                coordinate_map[(chain, seqidx)] = fasta_position

        if len([key for key in coordinate_map if key[0] == chain]) != len(by_index):
            raise ValueError(f"{path}: failed to align every {chain} SEQIDX to FASTA")
    return coordinate_map


def discover_systems(root: Path) -> list[SystemSpec]:
    if not root.is_dir():
        raise FileNotFoundError(f"system-bundle root not found: {root}")

    specs: list[SystemSpec] = []
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pdb_id = source_dir.name
        fasta_path = source_dir / f"{pdb_id}.fasta"
        mapping_path = source_dir / f"{pdb_id}.mapping"
        curated = sorted(source_dir.glob(f"{pdb_id}_*_New.txt"))
        if len(curated) != 1:
            raise ValueError(
                f"{source_dir}: expected one curated _New.txt table, found {len(curated)}"
            )
        if not fasta_path.is_file():
            raise FileNotFoundError(f"missing FASTA: {fasta_path}")
        if not mapping_path.is_file():
            raise FileNotFoundError(f"missing mapping: {mapping_path}")

        suffix = curated[0].name.removeprefix(f"{pdb_id}_").removesuffix("_New.txt")
        parts = suffix.split("_")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"cannot parse partner groups from {curated[0].name}")
        group1 = tuple(parts[0])
        group2 = tuple(parts[1])
        sequences = read_fasta(fasta_path)
        coordinate_map = read_mapping_positions(mapping_path, sequences)
        missing = set(group1 + group2) - set(sequences)
        if missing:
            raise ValueError(f"{fasta_path}: partner chains absent from FASTA: {missing}")

        specs.append(
            SystemSpec(
                pdb_id=pdb_id,
                name=f"{pdb_id}_{parts[0]}_{parts[1]}",
                source_dir=source_dir,
                curated_path=curated[0],
                fasta_path=fasta_path,
                group1=group1,
                group2=group2,
                sequences=sequences,
                seqidx_to_fasta_position=coordinate_map,
            )
        )
    if not specs:
        raise ValueError(f"no system directories found under {root}")
    return specs


def parse_mutations(label: str, *, context: str) -> tuple[Mutation, ...]:
    if label == "WT":
        return ()

    mutations: list[Mutation] = []
    occupied: dict[tuple[str, int], Mutation] = {}
    for raw_token in label.split(","):
        token = raw_token.strip()
        match = MUTATION_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"{context}: invalid mutation token {token!r}")
        mutation = Mutation(
            chain=match.group("chain"),
            position=int(match.group("position")),
            wt=match.group("wt"),
            mut=match.group("mut"),
        )
        key = (mutation.chain, mutation.position)
        if key in occupied:
            raise ValueError(
                f"{context}: position {mutation.chain}:{mutation.position} "
                "is mutated more than once"
            )
        if mutation.wt == mutation.mut:
            raise ValueError(f"{context}: no-op substitution {token!r}")
        occupied[key] = mutation
        mutations.append(mutation)
    return tuple(sorted(mutations))


def apply_mutations(
    sequences: dict[str, str],
    included_chains: Sequence[str],
    seqidx_to_fasta_position: dict[tuple[str, int], int],
    mutations: Sequence[Mutation],
    *,
    context: str,
) -> dict[str, str]:
    mutable = {chain: list(sequences[chain]) for chain in included_chains}
    for mutation in mutations:
        if mutation.chain not in mutable:
            raise ValueError(
                f"{context}: mutation chain {mutation.chain!r} is not in measured partners"
            )
        sequence = mutable[mutation.chain]
        coordinate_key = (mutation.chain, mutation.position)
        if coordinate_key not in seqidx_to_fasta_position:
            raise ValueError(
                f"{context}: no mapping SEQIDX for "
                f"{mutation.chain}:{mutation.position}"
            )
        fasta_position = seqidx_to_fasta_position[coordinate_key]
        index = fasta_position - 1
        if index >= len(sequence):
            raise ValueError(
                f"{context}: aligned FASTA position {fasta_position} exceeds length "
                f"{len(sequence)}"
            )
        observed = sequence[index]
        if observed != mutation.wt:
            raise ValueError(
                f"{context}: {mutation.text} expects {mutation.wt} but FASTA has "
                f"{observed} at {mutation.chain}:{fasta_position} "
                f"(mapping SEQIDX {mutation.position})"
            )
        sequence[index] = mutation.mut
    return {chain: "".join(mutable[chain]) for chain in included_chains}


def read_measurements(spec: SystemSpec) -> list[dict[str, str]]:
    with spec.curated_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle, delimiter="\t")
            if DATA_ID_RE.fullmatch(row.get("ID", ""))
        ]
    if not rows:
        raise ValueError(f"{spec.curated_path}: no data rows")
    wt_rows = [row for row in rows if row["Mutation"] == "WT"]
    if len(wt_rows) != 1:
        raise ValueError(
            f"{spec.curated_path}: expected one WT row, found {len(wt_rows)}"
        )

    wt_dg = float(wt_rows[0]["DG"])
    for row in rows:
        context = f"{spec.curated_path}:{row['ID']}"
        mutations = parse_mutations(row["Mutation"], context=context)
        apply_mutations(
            spec.sequences,
            spec.included_chains,
            spec.seqidx_to_fasta_position,
            mutations,
            context=context,
        )
        if mutations:
            ddg = float(row["DGmut"]) - float(row["DGWT"])
            expected_dg = wt_dg + ddg
            if abs(float(row["DG"]) - expected_dg) > 2e-4:
                raise ValueError(
                    f"{context}: DG re-anchoring mismatch: {row['DG']} vs {expected_dg}"
                )
    return rows


def canonical_label(mutations: Sequence[Mutation]) -> str:
    return "WT" if not mutations else ",".join(mutation.text for mutation in mutations)


def input_id(spec: SystemSpec, mutations: Sequence[Mutation]) -> str:
    if not mutations:
        return f"{spec.pdb_id}_WT"
    return f"{spec.pdb_id}_{'__'.join(mutation.text for mutation in mutations)}"


def render_yaml(sequences: dict[str, str], binder: str | None) -> str:
    lines = ["version: 1", "sequences:"]
    for chain, sequence in sequences.items():
        lines.extend(
            [
                "  - protein:",
                f"      id: {chain}",
                f"      sequence: {sequence}",
            ]
        )
    if binder is not None:
        lines.extend(
            [
                "properties:",
                "  - affinity:",
                f"      binder: {binder}",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_binder_overrides(values: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --binder-override {value!r}; use SYSTEM=CHAIN")
        system, chain = value.split("=", 1)
        if not system or len(chain) != 1:
            raise ValueError(f"invalid --binder-override {value!r}; use SYSTEM=CHAIN")
        if system in overrides:
            raise ValueError(f"duplicate binder override for {system}")
        overrides[system] = chain
    return overrides


def select_binder(
    spec: SystemSpec,
    side: str,
    overrides: dict[str, str],
) -> str | None:
    override = overrides.get(spec.name) or overrides.get(spec.pdb_id)
    if override is not None:
        if override not in spec.included_chains:
            raise ValueError(
                f"{spec.name}: binder override {override!r} is not a measured partner chain"
            )
        return override
    if side == "none":
        return None
    chains = spec.group1 if side == "group1" else spec.group2
    if len(chains) != 1:
        raise ValueError(
            f"{spec.name}: {side} is the multi-chain partner {''.join(chains)!r}; "
            "stock Boltz affinity accepts one chain and no scientifically neutral "
            "single-chain substitute exists. Supply --binder-override only if the "
            "custom runner's intended binder is known."
        )
    return chains[0]


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8g}"


def write_tsv(path: Path, rows: list[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def generate_system(
    spec: SystemSpec,
    out_root: Path,
    affinity_side: str,
    binder_overrides: dict[str, str],
) -> dict[str, object]:
    rows = read_measurements(spec)
    binder = select_binder(spec, affinity_side, binder_overrides)
    variants: dict[tuple[Mutation, ...], list[dict[str, str]]] = {}
    for row in rows:
        mutations = parse_mutations(
            row["Mutation"], context=f"{spec.curated_path}:{row['ID']}"
        )
        variants.setdefault(mutations, []).append(row)

    system_out = out_root / spec.name
    input_dir = system_out / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    desired_yaml_names: set[str] = set()
    measurement_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []

    ordered_variants = sorted(variants, key=lambda item: (bool(item), item))
    for mutations in ordered_variants:
        source_rows = variants[mutations]
        variant_id = input_id(spec, mutations)
        mutated_sequences = apply_mutations(
            spec.sequences,
            spec.included_chains,
            spec.seqidx_to_fasta_position,
            mutations,
            context=f"{spec.name}:{variant_id}",
        )
        yaml_name = f"{variant_id}.yaml"
        desired_yaml_names.add(yaml_name)
        (input_dir / yaml_name).write_text(
            render_yaml(mutated_sequences, binder), encoding="utf-8"
        )

        ddgs = [
            0.0
            if row["Mutation"] == "WT"
            else float(row["DGmut"]) - float(row["DGWT"])
            for row in source_rows
        ]
        dgs = [float(row["DG"]) for row in source_rows]
        variant_rows.append(
            {
                "input_id": variant_id,
                "system": spec.name,
                "pdb_id": spec.pdb_id,
                "mutation": canonical_label(mutations),
                "n_substitutions": len(mutations),
                "n_measurements": len(source_rows),
                "measurement_ids": ",".join(row["ID"] for row in source_rows),
                "ddg_median_kcal_mol": format_float(statistics.median(ddgs)),
                "ddg_mean_kcal_mol": format_float(statistics.fmean(ddgs)),
                "ddg_sd_kcal_mol": format_float(
                    statistics.stdev(ddgs) if len(ddgs) > 1 else None
                ),
                "ddg_min_kcal_mol": format_float(min(ddgs)),
                "ddg_max_kcal_mol": format_float(max(ddgs)),
                "dg_median_kcal_mol": format_float(statistics.median(dgs)),
                "group1_chains": "".join(spec.group1),
                "group2_chains": "".join(spec.group2),
                "included_chains": ",".join(spec.included_chains),
                "excluded_fasta_chains": ",".join(spec.excluded_chains),
                "mapping_adjusted_chains": ",".join(spec.mapping_adjusted_chains),
                "affinity_binder": binder or "",
                "chain_sequences_json": json.dumps(
                    mutated_sequences, separators=(",", ":"), sort_keys=True
                ),
            }
        )
        for row, ddg in zip(source_rows, ddgs):
            measurement_rows.append(
                {
                    "measurement_id": row["ID"],
                    "input_id": variant_id,
                    "system": spec.name,
                    "pdb_id": spec.pdb_id,
                    "mutation": row["Mutation"],
                    "canonical_mutation": canonical_label(mutations),
                    "n_substitutions": len(mutations),
                    "DG": row["DG"],
                    "Activity_Mutate": row["Activity_Mutate"],
                    "DGmut": row["DGmut"],
                    "Activity_WT": row["Activity_WT"],
                    "DGWT": row["DGWT"],
                    "ddg_kcal_mol": format_float(ddg),
                }
            )

    # Remove stale YAMLs from an earlier generation without touching any other file.
    for stale_path in input_dir.glob("*.yaml"):
        if stale_path.name not in desired_yaml_names:
            stale_path.unlink()

    write_tsv(
        system_out / "variants.tsv",
        variant_rows,
        (
            "input_id",
            "system",
            "pdb_id",
            "mutation",
            "n_substitutions",
            "n_measurements",
            "measurement_ids",
            "ddg_median_kcal_mol",
            "ddg_mean_kcal_mol",
            "ddg_sd_kcal_mol",
            "ddg_min_kcal_mol",
            "ddg_max_kcal_mol",
            "dg_median_kcal_mol",
            "group1_chains",
            "group2_chains",
            "included_chains",
            "excluded_fasta_chains",
            "mapping_adjusted_chains",
            "affinity_binder",
            "chain_sequences_json",
        ),
    )
    write_tsv(
        system_out / "measurements.tsv",
        measurement_rows,
        (
            "measurement_id",
            "input_id",
            "system",
            "pdb_id",
            "mutation",
            "canonical_mutation",
            "n_substitutions",
            "DG",
            "Activity_Mutate",
            "DGmut",
            "Activity_WT",
            "DGWT",
            "ddg_kcal_mol",
        ),
    )

    mutant_rows = [row for row in rows if row["Mutation"] != "WT"]
    mutant_variants = [mutations for mutations in variants if mutations]
    return {
        "system": spec.name,
        "pdb_id": spec.pdb_id,
        "group1_chains": "".join(spec.group1),
        "group2_chains": "".join(spec.group2),
        "included_chains": ",".join(spec.included_chains),
        "excluded_fasta_chains": ",".join(spec.excluded_chains),
        "mapping_adjusted_chains": ",".join(spec.mapping_adjusted_chains),
        "affinity_binder": binder or "",
        "measurement_rows": len(mutant_rows),
        "unique_mutants": len(mutant_variants),
        "inputs_with_wt": len(variants),
        "repeated_measurement_rows": len(mutant_rows) - len(mutant_variants),
        "mutants_with_repeated_measurements": sum(
            len(variants[mutations]) > 1 for mutations in mutant_variants
        ),
        "multimutant_inputs": sum(len(mutations) > 1 for mutations in mutant_variants),
        "max_substitutions": max(map(len, mutant_variants)),
        "wt_reference_dg_kcal_mol": next(
            row["DG"] for row in rows if row["Mutation"] == "WT"
        ),
    }


def main() -> int:
    args = parse_args()
    specs = discover_systems(args.systems_root.resolve())
    if args.systems:
        requested = set(args.systems)
        specs = [
            spec
            for spec in specs
            if spec.pdb_id in requested or spec.name in requested
        ]
        found = {spec.pdb_id for spec in specs} | {spec.name for spec in specs}
        missing = requested - found
        if missing:
            raise ValueError(f"requested systems not found: {sorted(missing)}")

    overrides = parse_binder_overrides(args.binder_override)
    known = {spec.pdb_id for spec in specs} | {spec.name for spec in specs}
    unknown_overrides = set(overrides) - known
    if unknown_overrides:
        raise ValueError(f"binder overrides reference unknown systems: {unknown_overrides}")

    out_root = args.out_root.resolve()
    summaries = [
        generate_system(spec, out_root, args.affinity_side, overrides)
        for spec in specs
    ]
    write_tsv(
        out_root / "manifest.tsv",
        summaries,
        (
            "system",
            "pdb_id",
            "group1_chains",
            "group2_chains",
            "included_chains",
            "excluded_fasta_chains",
            "mapping_adjusted_chains",
            "affinity_binder",
            "measurement_rows",
            "unique_mutants",
            "inputs_with_wt",
            "repeated_measurement_rows",
            "mutants_with_repeated_measurements",
            "multimutant_inputs",
            "max_substitutions",
            "wt_reference_dg_kcal_mol",
        ),
    )

    total_measurements = sum(int(row["measurement_rows"]) for row in summaries)
    total_mutants = sum(int(row["unique_mutants"]) for row in summaries)
    total_inputs = sum(int(row["inputs_with_wt"]) for row in summaries)
    print(
        f"Generated {total_inputs} YAMLs ({total_mutants} unique mutants + "
        f"{len(summaries)} WT) from {total_measurements} mutant measurement rows."
    )
    print(f"Manifests and inputs: {out_root}")
    if args.affinity_side == "none" and not overrides:
        print(
            "Affinity property omitted (structure-only inputs; choose an MSA policy "
            "before stock-Boltz execution)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
