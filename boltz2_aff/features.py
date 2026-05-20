"""Discover Boltz affinity embeddings, scalar affinity predictions, and ligand fingerprints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import ConvertToNumpyArray
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False

EMBEDDING_PREFIX = "affinity_embeddings_"
EMBEDDING_KEY_CHOICES: tuple[str, ...] = ("pair_mean1", "head1", "pair_mean2", "head2")
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def _normalize_embedding_keys(keys: Iterable[str] | None) -> set[str] | None:
    if keys is None:
        return None
    cleaned = {key.strip().lower() for key in keys if key and key.strip()}
    unknown = cleaned - set(EMBEDDING_KEY_CHOICES)
    if unknown:
        raise ValueError(
            f"unknown embedding keys: {sorted(unknown)}. choose from {list(EMBEDDING_KEY_CHOICES)}"
        )
    return cleaned or None


def _embedding_key_short_name(key: str) -> str:
    base = "pair_mean" if "pair_mean" in key else "head" if "head" in key else None
    if base is None:
        return ""
    suffix_match = re.search(r"(\d+)$", key)
    suffix = suffix_match.group(1) if suffix_match else ""
    return f"{base}{suffix}"


def _target_filter(targets: Iterable[str] | None) -> set[str] | None:
    return {target.upper() for target in targets} if targets else None


def _variant_filter(variants: Iterable[str] | None) -> set[str] | None:
    return {variant.lower() for variant in variants} if variants else None


def ligand_id_from_embedding(path: Path) -> str:
    name = path.stem
    if name.startswith(EMBEDDING_PREFIX):
        return name[len(EMBEDDING_PREFIX) :]
    return name


def ligand_id_from_affinity_json(path: Path) -> str:
    name = path.stem
    if name.startswith("affinity_"):
        return name[len("affinity_") :]
    return name


def _infer_layout(root: Path, path: Path, ligand_id: str) -> tuple[str, str]:
    """Infer target and variant from supported embedding/output layouts."""

    rel = path.relative_to(root)
    parts = rel.parts

    # data/Boltz-2/<target>/<variant>/output/<ligand>/affinity_embeddings_<ligand>.npz
    if len(parts) >= 5 and parts[2].lower() == "output":
        return parts[0], parts[1]

    # targets/<target>/affinity_embeddings_<ligand>.npz
    if len(parts) >= 2:
        return parts[0], "wt"

    # <target dir>/affinity_embeddings_<ligand>.npz
    return root.name, "wt"


def _embedding_key_order(key: str) -> tuple[int, int, str]:
    base_order = 0 if "pair_mean" in key else 1 if "head" in key else 2
    suffix_match = re.search(r"(\d+)$", key)
    suffix = int(suffix_match.group(1)) if suffix_match else 0
    return suffix, base_order, key


def _flatten_embedding(path: Path, embedding_keys: set[str] | None = None) -> dict[str, float]:
    arrays = np.load(path)
    features: dict[str, float] = {}
    for key in sorted(arrays.files, key=_embedding_key_order):
        short = _embedding_key_short_name(key)
        if embedding_keys is not None and short not in embedding_keys:
            continue
        array = np.asarray(arrays[key], dtype=np.float32)
        if array.ndim >= 2 and array.shape[0] > 1:
            array = array.mean(axis=0)
        array = np.squeeze(array)
        flat = array.reshape(-1)
        safe_key = re.sub(r"[^0-9A-Za-z_]+", "_", key)
        for index, value in enumerate(flat):
            features[f"emb_{safe_key}_{index:04d}"] = float(value)
    return features


def discover_embedding_frame(
    roots: Iterable[Path],
    targets: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
    embedding_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load all affinity embedding npz files under supported roots."""

    target_filter = _target_filter(targets)
    variant_filter = _variant_filter(variants)
    key_filter = _normalize_embedding_keys(embedding_keys)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(f"{EMBEDDING_PREFIX}*.npz")):
            ligand_id = ligand_id_from_embedding(path)
            target, variant = _infer_layout(root, path, ligand_id)
            if target_filter and target.upper() not in target_filter:
                continue
            if variant_filter and variant.lower() not in variant_filter:
                continue
            key = (target.upper(), variant.lower(), ligand_id)
            if key in seen:
                continue
            seen.add(key)
            row: dict[str, object] = {
                "target": target,
                "variant": variant,
                "ligand_id": ligand_id,
                "embedding_path": str(path),
            }
            row.update(_flatten_embedding(path, embedding_keys=key_filter))
            rows.append(row)

    return pd.DataFrame(rows)


def discover_boltz_scalar_frame(
    boltz_output_root: Path,
    targets: Iterable[str] | None = None,
    variants: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load scalar affinity JSON predictions from Boltz output folders."""

    if not boltz_output_root.exists():
        return pd.DataFrame()

    target_filter = _target_filter(targets)
    variant_filter = _variant_filter(variants)
    rows: list[dict[str, object]] = []

    for path in sorted(boltz_output_root.rglob("affinity_*.json")):
        if path.name.startswith("confidence_"):
            continue
        ligand_id = ligand_id_from_affinity_json(path)
        target, variant = _infer_layout(boltz_output_root, path, ligand_id)
        if target_filter and target.upper() not in target_filter:
            continue
        if variant_filter and variant.lower() not in variant_filter:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row: dict[str, object] = {
            "target": target,
            "variant": variant,
            "ligand_id": ligand_id,
            "boltz_affinity_json_path": str(path),
        }
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                safe_key = re.sub(r"[^0-9A-Za-z_]+", "_", key)
                row[f"boltz_{safe_key}"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def _mol2_blocks(mol2_text: str) -> list[str]:
    """Split a multi-molecule mol2 string into individual molecule blocks."""
    marker = "@<TRIPOS>MOLECULE"
    parts = mol2_text.split(marker)
    return [marker + part for part in parts[1:]]


def _mol2_ligand_name(block: str) -> str:
    """Extract the molecule name (first non-empty line after the MOLECULE header)."""
    lines = block.splitlines()
    for line in lines[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("@"):
            return stripped
    return ""


def _mol2_to_fingerprint(block: str) -> tuple[str, np.ndarray] | None:
    """Parse one mol2 block and return (ligand_id, ECFP4 bit array) or None on failure."""
    if not _RDKIT_AVAILABLE:
        return None
    name = _mol2_ligand_name(block)
    mol = Chem.MolFromMol2Block(block, sanitize=True, removeHs=True)
    if mol is None:
        mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=True)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return name, arr


def discover_ligand_fingerprint_frame(
    ulvsh_root: Path,
    targets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute ECFP4 Morgan fingerprints from ULVSH poses.mol2 files.

    Returns a DataFrame with columns target, ligand_id, lig_ecfp4_0000 … lig_ecfp4_2047.
    Returns an empty DataFrame if rdkit is unavailable or no mol2 files are found.
    """
    if not _RDKIT_AVAILABLE:
        return pd.DataFrame()

    target_filter = _target_filter(targets)
    rows: list[dict[str, object]] = []

    for mol2_path in sorted(ulvsh_root.rglob("poses.mol2")):
        # Expected layout: ulvsh_root/<target>/raw/poses.mol2
        target = mol2_path.parent.parent.name
        if target_filter and target.upper() not in target_filter:
            continue
        mol2_text = mol2_path.read_text(encoding="utf-8", errors="replace")
        n_ok = n_fail = 0
        for block in _mol2_blocks(mol2_text):
            result = _mol2_to_fingerprint(block)
            if result is None:
                n_fail += 1
                continue
            ligand_id, fp_arr = result
            if not ligand_id:
                n_fail += 1
                continue
            row: dict[str, object] = {"target": target, "ligand_id": ligand_id}
            for i, bit in enumerate(fp_arr):
                row[f"lig_ecfp4_{i:04d}"] = int(bit)
            rows.append(row)
            n_ok += 1
        if n_fail:
            import warnings
            warnings.warn(
                f"discover_ligand_fingerprint_frame: {target} — {n_fail} mol2 blocks failed to parse"
                f" ({n_ok} ok)",
                stacklevel=2,
            )

    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    prefixes_by_set = {
        "embeddings": ("emb_",),
        "boltz": ("boltz_",),
        "ulvsh_scores": ("score_",),
        "combined": ("emb_", "boltz_", "score_"),
        "ligand": ("lig_",),
        "combined_ligand": ("emb_", "boltz_", "score_", "lig_"),
    }
    prefixes = prefixes_by_set[feature_set]
    columns = [column for column in frame.columns if column.startswith(prefixes)]
    usable = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            usable.append(column)
    return usable
