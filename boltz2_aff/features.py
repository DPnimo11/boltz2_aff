"""Discover Boltz affinity embeddings and scalar affinity predictions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EMBEDDING_PREFIX = "affinity_embeddings_"


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


def _flatten_embedding(path: Path) -> dict[str, float]:
    arrays = np.load(path)
    features: dict[str, float] = {}
    for key in sorted(arrays.files, key=_embedding_key_order):
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
) -> pd.DataFrame:
    """Load all affinity embedding npz files under supported roots."""

    target_filter = _target_filter(targets)
    variant_filter = _variant_filter(variants)
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
            row.update(_flatten_embedding(path))
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


def feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    prefixes_by_set = {
        "embeddings": ("emb_",),
        "boltz": ("boltz_",),
        "ulvsh_scores": ("score_",),
        "combined": ("emb_", "boltz_", "score_"),
    }
    prefixes = prefixes_by_set[feature_set]
    columns = [column for column in frame.columns if column.startswith(prefixes)]
    usable = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            usable.append(column)
    return usable
