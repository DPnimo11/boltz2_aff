"""Load and normalize ULVSH affinity labels and optional score features."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_COLUMNS = {"ID", "Active"}


def _read_whitespace_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _find_column(columns: Iterable[str], wanted: str) -> str | None:
    wanted_norm = wanted.lower().replace("_", "").replace("(", "").replace(")", "")
    for column in columns:
        norm = column.lower().replace("_", "").replace("(", "").replace(")", "")
        if norm == wanted_norm:
            return column
    return None


def _find_affinity_column(columns: Iterable[str]) -> str | None:
    """Find a quantitative affinity column, ignoring percent activity columns."""

    for column in columns:
        normalized = column.lower().replace("_", "").replace("(", "").replace(")", "")
        if normalized == "pki":
            return column
    for column in columns:
        normalized = column.lower()
        if normalized.startswith("ki") or "ec50" in normalized or "ic50" in normalized or normalized.startswith("kd"):
            return column
    return None


def _parse_active(value: object) -> bool | float:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "active"}:
        return True
    if text in {"no", "n", "false", "0", "inactive"}:
        return False
    return np.nan


def _parse_affinity_value(value: object) -> tuple[float, bool]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none"}:
        return np.nan, False

    is_censored = text.startswith("<") or text.startswith(">") or "%" in text
    if "%" in text:
        return np.nan, is_censored

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return np.nan, is_censored

    value_float = float(match.group(0))
    if value_float <= 0 or not np.isfinite(value_float):
        return np.nan, is_censored
    if is_censored:
        return np.nan, True
    return value_float, False


def _measurement_to_um(value: float, column: str) -> float:
    normalized = column.lower().replace(" ", "")
    if "nm" in normalized:
        return value / 1000.0
    return value


def _p_affinity_from_column(value: float, column: str) -> tuple[float, float]:
    normalized = column.lower().replace("_", "").replace("(", "").replace(")", "")
    if normalized == "pki":
        p_affinity = value
        affinity_um = float(np.power(10.0, 6.0 - p_affinity))
    else:
        affinity_um = _measurement_to_um(value, column)
        p_affinity = float(6.0 - np.log10(affinity_um))
    return affinity_um, p_affinity


def _is_label_like_column(column: str) -> bool:
    normalized = column.lower()
    return (
        column in LABEL_COLUMNS
        or normalized.startswith("ki")
        or normalized == "pki"
        or "ec50" in normalized
        or "ic50" in normalized
        or normalized.startswith("kd")
        or normalized.startswith("%")
    )


def _clean_score_features(scores: pd.DataFrame) -> pd.DataFrame:
    id_column = _find_column(scores.columns, "ID")
    if id_column is None:
        raise ValueError("scores table is missing an ID column")

    scores = scores.rename(columns={id_column: "ligand_id"}).copy()
    drop_columns = [col for col in scores.columns if _is_label_like_column(col) or col == "ligand_id"]
    feature_columns = [col for col in scores.columns if col not in drop_columns]

    cleaned = scores[["ligand_id"]].copy()
    for column in feature_columns:
        output_column = f"score_{column}"
        text = scores[column].astype(str).str.strip().str.lower()
        yes_no = text.map({"yes": 1.0, "no": 0.0})
        numeric = pd.to_numeric(scores[column], errors="coerce")
        cleaned[output_column] = numeric.where(numeric.notna(), yes_no)
    return cleaned


def load_ulvsh_target(
    target_dir: Path,
    score_source: str = "raw",
    include_scores: bool = True,
) -> pd.DataFrame:
    """Load one ULVSH target directory into normalized modeling rows."""

    target = target_dir.name
    vitro_path = target_dir / "raw" / "vitro.tsv"
    if not vitro_path.exists():
        raise FileNotFoundError(f"missing ULVSH label file: {vitro_path}")

    vitro = _read_whitespace_table(vitro_path)
    id_column = _find_column(vitro.columns, "ID")
    active_column = _find_column(vitro.columns, "Active")
    affinity_column = _find_affinity_column(vitro.columns)
    if id_column is None or active_column is None:
        raise ValueError(f"{vitro_path} must contain ID and Active columns")

    labels = pd.DataFrame(
        {
            "target": target,
            "ligand_id": vitro[id_column].astype(str),
            "active_raw": vitro[active_column].astype(str),
        }
    )
    labels["affinity_source"] = affinity_column or ""
    if affinity_column is None:
        labels["affinity_raw"] = ""
        labels["affinity_um"] = np.nan
        labels["p_affinity"] = np.nan
        labels["affinity_is_censored"] = False
    else:
        labels["affinity_raw"] = vitro[affinity_column].astype(str)
        parsed_affinity = labels["affinity_raw"].map(_parse_affinity_value)
        values = [item[0] for item in parsed_affinity]
        labels["affinity_is_censored"] = [item[1] for item in parsed_affinity]
        converted = [
            _p_affinity_from_column(value, affinity_column) if pd.notna(value) else (np.nan, np.nan)
            for value in values
        ]
        labels["affinity_um"] = [item[0] for item in converted]
        labels["p_affinity"] = [item[1] for item in converted]

    # Backward-compatible aliases for Ki-style analyses and quick notebooks.
    labels["ki_raw"] = labels["affinity_raw"]
    labels["ki_um"] = labels["affinity_um"]
    labels["ki_is_censored"] = labels["affinity_is_censored"]
    labels["pki"] = labels["p_affinity"]
    labels["active_bool"] = labels["active_raw"].map(_parse_active)

    if include_scores:
        scores_path = target_dir / score_source / "scores.tsv"
        if scores_path.exists():
            scores = _clean_score_features(_read_whitespace_table(scores_path))
            labels = labels.merge(scores, on="ligand_id", how="left")

    return labels


def load_ulvsh(
    ulvsh_root: Path,
    targets: Iterable[str] | None = None,
    score_source: str = "raw",
    include_scores: bool = True,
) -> pd.DataFrame:
    """Load all requested ULVSH targets."""

    target_filter = {target.upper() for target in targets} if targets else None
    frames: list[pd.DataFrame] = []
    for target_dir in sorted(path for path in ulvsh_root.iterdir() if path.is_dir()):
        if target_filter and target_dir.name.upper() not in target_filter:
            continue
        frames.append(load_ulvsh_target(target_dir, score_source, include_scores))

    if not frames:
        requested = ", ".join(sorted(target_filter)) if target_filter else "all targets"
        raise ValueError(f"no ULVSH targets loaded from {ulvsh_root} for {requested}")
    return pd.concat(frames, ignore_index=True)
