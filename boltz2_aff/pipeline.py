"""Command line pipeline for ULVSH + Boltz-2 affinity embedding modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from boltz2_aff.data import load_ulvsh
from boltz2_aff.features import (
    discover_boltz_scalar_frame,
    discover_embedding_frame,
    feature_columns,
)
from boltz2_aff.modeling import train_classifier, train_regressor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ulvsh-root", type=Path, default=Path("data/ULVSH"))
    parser.add_argument(
        "--embedding-root",
        type=Path,
        action="append",
        default=[Path("targets")],
        help="Root containing affinity_embeddings_*.npz files. Can be repeated.",
    )
    parser.add_argument("--boltz-output-root", type=Path, default=Path("data/Boltz-2"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--targets", nargs="*", default=None, help="Optional ULVSH target names.")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional Boltz variants, e.g. wt mut shuffled.")
    parser.add_argument("--score-source", default="raw", choices=["raw", "minimized"])
    parser.add_argument(
        "--feature-set",
        default="embeddings",
        choices=["embeddings", "boltz", "ulvsh_scores", "combined"],
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["classification", "regression"],
        choices=["classification", "regression"],
    )
    parser.add_argument("--max-cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _merge_features(
    labels: pd.DataFrame,
    embeddings: pd.DataFrame,
    boltz_scalars: pd.DataFrame,
    feature_set: str,
) -> pd.DataFrame:
    needs_embeddings = feature_set in {"embeddings", "combined"}
    needs_boltz = feature_set in {"boltz", "combined"}

    if needs_embeddings:
        if embeddings.empty:
            raise ValueError("no affinity embedding files were found for the requested selection")
        frame = labels.merge(embeddings, on=["target", "ligand_id"], how="inner")
    elif needs_boltz:
        if boltz_scalars.empty:
            raise ValueError("no Boltz affinity JSON files were found for the requested selection")
        frame = labels.merge(boltz_scalars, on=["target", "ligand_id"], how="inner")
    else:
        frame = labels.copy()
        frame["variant"] = "ulvsh"

    if needs_boltz and not boltz_scalars.empty and needs_embeddings:
        scalar_columns = [
            column
            for column in boltz_scalars.columns
            if column.startswith("boltz_") or column == "boltz_affinity_json_path"
        ]
        frame = frame.merge(
            boltz_scalars[["target", "variant", "ligand_id", *scalar_columns]],
            on=["target", "variant", "ligand_id"],
            how="left",
        )

    return frame


def _missing_embedding_summary(labels: pd.DataFrame, embeddings: pd.DataFrame) -> dict[str, Any]:
    label_keys = labels[["target", "ligand_id"]].drop_duplicates()
    if embeddings.empty:
        return {
            "labels_without_embeddings": int(len(label_keys)),
            "targets_with_embeddings": [],
        }
    embedding_keys = embeddings[["target", "ligand_id"]].drop_duplicates()
    merged = label_keys.merge(embedding_keys, on=["target", "ligand_id"], how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]
    return {
        "labels_without_embeddings": int(len(missing)),
        "targets_with_embeddings": sorted(embeddings["target"].dropna().unique().tolist()),
        "missing_by_target": missing.groupby("target").size().astype(int).to_dict(),
    }


def _write_manifest(
    out_dir: Path,
    labels: pd.DataFrame,
    embeddings: pd.DataFrame,
    boltz_scalars: pd.DataFrame,
    frame: pd.DataFrame,
    selected_features: list[str],
    embedding_roots: list[Path],
    args: argparse.Namespace,
    task_metrics: dict[str, Any],
) -> None:
    manifest = {
        "ulvsh_root": str(args.ulvsh_root),
        "embedding_roots": [str(path) for path in embedding_roots],
        "boltz_output_root": str(args.boltz_output_root),
        "feature_set": args.feature_set,
        "targets_requested": args.targets,
        "variants_requested": args.variants,
        "score_source": args.score_source,
        "n_label_rows": int(len(labels)),
        "n_embedding_rows": int(len(embeddings)),
        "n_boltz_scalar_rows": int(len(boltz_scalars)),
        "n_modeling_rows": int(len(frame)),
        "n_features": int(len(selected_features)),
        "target_counts": frame.groupby("target").size().astype(int).to_dict(),
        "variant_counts": frame.groupby("variant").size().astype(int).to_dict(),
        "embedding_coverage": _missing_embedding_summary(labels, embeddings),
        "metrics": task_metrics,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    embedding_roots = list(dict.fromkeys([*args.embedding_root, args.boltz_output_root]))
    labels = load_ulvsh(args.ulvsh_root, args.targets, args.score_source, include_scores=True)
    embeddings = discover_embedding_frame(embedding_roots, args.targets, args.variants)
    boltz_scalars = discover_boltz_scalar_frame(args.boltz_output_root, args.targets, args.variants)
    frame = _merge_features(labels, embeddings, boltz_scalars, args.feature_set)
    frame["group_id"] = frame["target"].astype(str) + "::" + frame["ligand_id"].astype(str)

    selected_features = feature_columns(frame, args.feature_set)
    if not selected_features:
        raise ValueError(f"no usable numeric features found for feature set {args.feature_set!r}")

    metadata_columns = [
        "target",
        "variant",
        "ligand_id",
        "group_id",
        "affinity_source",
        "affinity_raw",
        "affinity_um",
        "affinity_is_censored",
        "p_affinity",
        "ki_raw",
        "ki_um",
        "ki_is_censored",
        "pki",
        "active_raw",
        "active_bool",
        "embedding_path",
        "boltz_affinity_json_path",
    ]
    ordered_columns = [column for column in metadata_columns if column in frame.columns]
    ordered_columns += [column for column in selected_features if column in frame.columns]
    frame[ordered_columns].to_csv(out_dir / "dataset.csv", index=False)

    task_metrics: dict[str, Any] = {}
    if "classification" in args.tasks:
        task_metrics["classification"] = train_classifier(
            frame,
            selected_features,
            model_dir,
            max_splits=args.max_cv_splits,
            random_state=args.random_state,
        )
    if "regression" in args.tasks:
        task_metrics["regression"] = train_regressor(
            frame,
            selected_features,
            model_dir,
            max_splits=args.max_cv_splits,
        )

    _write_manifest(
        out_dir,
        labels,
        embeddings,
        boltz_scalars,
        frame,
        selected_features,
        embedding_roots,
        args,
        task_metrics,
    )
    return task_metrics


def main() -> None:
    args = _parse_args()
    metrics = run(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
