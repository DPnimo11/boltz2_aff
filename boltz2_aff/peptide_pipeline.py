"""Per-system nested-CV modeling for the SKEMPI peptide-system dataset."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODELING_ROOT = Path("data/peptide_systems/modeling")
DEFAULT_OUT_DIR = Path("runs/peptide_systems/ridge")
DEFAULT_VIEWS = (
    "mutation",
    "pair_mean",
    "mutation+pair_mean",
    "head_mean",
    "mutation+head_mean",
)
EMBEDDING_KEYS = ("pair_mean", "head_ens1", "head_ens2", "head_mean")
LABEL_COLUMNS = {
    "median": "ddg_median_kcal_mol",
    "mean": "ddg_mean_kcal_mol",
}
_MUTATION_TOKEN = re.compile(r"^([A-Z])([A-Za-z0-9])(\d+)([A-Z])$")


@dataclass(frozen=True)
class PeptideDataset:
    """Validated labels, observations, and aligned embedding blocks."""

    labels: pd.DataFrame
    measurements: pd.DataFrame
    embeddings: dict[str, np.ndarray]


def _required_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"missing peptide-system modeling file: {path}")
    return path


def load_peptide_dataset(root: Path = DEFAULT_MODELING_ROOT) -> PeptideDataset:
    """Load the modeling dataset and enforce its join/order invariants."""

    labels = pd.read_csv(_required_path(root, "labels.tsv"), sep="\t")
    measurements = pd.read_csv(_required_path(root, "measurements.tsv"), sep="\t")
    index = pd.read_csv(_required_path(root, "index.tsv"), sep="\t")
    archive_path = _required_path(root, "features/boltz_embeddings.npz")

    required_labels = {
        "system",
        "input_id",
        "mutation",
        "n_substitutions",
        "n_measurements",
        "ddg_median_kcal_mol",
        "ddg_mean_kcal_mol",
    }
    missing_labels = required_labels.difference(labels.columns)
    if missing_labels:
        raise ValueError(f"labels.tsv is missing columns: {sorted(missing_labels)}")
    if labels.duplicated(["system", "input_id"]).any():
        raise ValueError("labels.tsv has duplicate (system, input_id) keys")

    archive = np.load(archive_path)
    missing_arrays = {"ids", "target", *EMBEDDING_KEYS}.difference(archive.files)
    if missing_arrays:
        raise ValueError(f"embedding archive is missing arrays: {sorted(missing_arrays)}")

    ids = np.asarray(archive["ids"]).astype(str)
    targets = np.asarray(archive["target"]).astype(str)
    if len(ids) != len(targets) or len(set(ids.tolist())) != len(ids):
        raise ValueError("embedding IDs/targets have inconsistent or duplicate rows")

    systems: list[str] = []
    input_ids: list[str] = []
    for composite_id, target in zip(ids, targets):
        system, separator, input_id = composite_id.partition("::")
        if not separator or system != target:
            raise ValueError(f"invalid embedding composite ID: {composite_id!r}")
        systems.append(system)
        input_ids.append(input_id)

    archive_index = pd.DataFrame(
        {"row": np.arange(len(ids)), "system": systems, "input_id": input_ids}
    )
    expected_index = index[["row", "system", "input_id"]].copy()
    expected_index["row"] = pd.to_numeric(expected_index["row"], errors="raise").astype(int)
    if not archive_index.equals(expected_index.reset_index(drop=True)):
        raise ValueError("index.tsv does not exactly reproduce embedding archive order")

    aligned = labels.merge(
        archive_index,
        on=["system", "input_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned["row"].isna().any() or len(aligned) != len(ids):
        raise ValueError("embedding and label key sets do not match exactly")
    aligned["row"] = aligned["row"].astype(int)

    measurement_keys = measurements[["system", "input_id"]].drop_duplicates()
    unmatched = measurement_keys.merge(
        labels[["system", "input_id"]],
        on=["system", "input_id"],
        how="left",
        indicator=True,
    )
    if (unmatched["_merge"] != "both").any():
        raise ValueError("measurements.tsv contains keys absent from labels.tsv")

    rows = aligned["row"].to_numpy()
    embedding_blocks: dict[str, np.ndarray] = {}
    for key in EMBEDDING_KEYS:
        block = np.asarray(archive[key], dtype=np.float64)
        if block.ndim != 2 or block.shape[0] != len(ids):
            raise ValueError(f"embedding array {key!r} has unexpected shape {block.shape}")
        block = block[rows]
        if not np.isfinite(block).all():
            raise ValueError(f"embedding array {key!r} contains non-finite values")
        embedding_blocks[key] = block

    return PeptideDataset(
        labels=aligned.drop(columns="row"),
        measurements=measurements,
        embeddings=embedding_blocks,
    )


def _parse_mutation(mutation: str) -> list[tuple[str, str, int, str]]:
    if mutation == "WT":
        return []
    parsed: list[tuple[str, str, int, str]] = []
    for token in mutation.split(","):
        match = _MUTATION_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"unsupported mutation token: {token!r}")
        wt, chain, position, mutant = match.groups()
        parsed.append((wt, chain, int(position), mutant))
    return parsed


def _mutation_features(rows: pd.DataFrame) -> tuple[np.ndarray, list[str], DictVectorizer]:
    records: list[dict[str, float]] = []
    for row in rows.itertuples(index=False):
        features: dict[str, float] = {"n_substitutions": float(row.n_substitutions)}
        for wt, chain, position, mutant in _parse_mutation(str(row.mutation)):
            features[f"position::{chain}{position}"] = 1.0
            features[f"substitution::{wt}>{mutant}"] = (
                features.get(f"substitution::{wt}>{mutant}", 0.0) + 1.0
            )
            features[f"position_substitution::{chain}{position}:{wt}>{mutant}"] = 1.0
        records.append(features)
    vectorizer = DictVectorizer(sparse=False, sort=True)
    matrix = np.asarray(vectorizer.fit_transform(records), dtype=np.float64)
    return matrix, vectorizer.get_feature_names_out().tolist(), vectorizer


def _delta_embedding(
    dataset: PeptideDataset,
    system_rows: pd.DataFrame,
    system_indices: np.ndarray,
    key: str,
) -> np.ndarray:
    wt_local = np.flatnonzero(system_rows["mutation"].astype(str).to_numpy() == "WT")
    if len(wt_local) != 1:
        raise ValueError(
            f"system {system_rows['system'].iloc[0]!r} must contain exactly one WT row"
        )
    block = dataset.embeddings[key][system_indices]
    return block - block[wt_local[0]]


def _system_feature_blocks(
    dataset: PeptideDataset, system: str
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, list[str]]], DictVectorizer]:
    system_mask = dataset.labels["system"].astype(str).to_numpy() == system
    system_indices = np.flatnonzero(system_mask)
    all_rows = dataset.labels.iloc[system_indices].reset_index(drop=True)
    mutant_mask = all_rows["mutation"].astype(str).to_numpy() != "WT"
    rows = all_rows.loc[mutant_mask].reset_index(drop=True)

    mutation, mutation_names, vectorizer = _mutation_features(rows)
    blocks: dict[str, tuple[np.ndarray, list[str]]] = {
        "mutation": (mutation, mutation_names)
    }
    for key in EMBEDDING_KEYS:
        delta = _delta_embedding(dataset, all_rows, system_indices, key)[mutant_mask]
        names = [f"delta_{key}_{column:04d}" for column in range(delta.shape[1])]
        blocks[key] = (delta, names)
    return rows, blocks, vectorizer


def _assemble_view(
    blocks: dict[str, tuple[np.ndarray, list[str]]], view: str
) -> tuple[np.ndarray, list[str]]:
    keys = view.split("+")
    unknown = [key for key in keys if key not in blocks]
    if unknown:
        raise ValueError(f"unknown feature blocks in view {view!r}: {unknown}")
    matrices = [blocks[key][0] for key in keys]
    names = [name for key in keys for name in blocks[key][1]]
    return np.concatenate(matrices, axis=1), names


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )


def _nested_ridge_oof(
    x: np.ndarray,
    y: np.ndarray,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], GridSearchCV]:
    if len(y) < 4:
        raise ValueError("nested CV requires at least four mutant rows")
    n_outer = min(outer_splits, len(y))
    if n_outer < 2:
        raise ValueError("nested CV requires at least two outer folds")

    alpha_grid = np.logspace(-4, 4, 33)
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    predictions = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_details: list[dict[str, Any]] = []

    for fold, (train, test) in enumerate(outer.split(x)):
        n_inner = min(inner_splits, len(train))
        if n_inner < 2:
            raise ValueError("nested CV requires at least two inner folds")
        inner = KFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=random_state + fold + 1,
        )
        search = GridSearchCV(
            _ridge_pipeline(),
            param_grid={"model__alpha": alpha_grid},
            scoring="neg_mean_absolute_error",
            cv=inner,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(x[train], y[train])
        predictions[test] = search.predict(x[test])
        fold_ids[test] = fold
        fold_details.append(
            {
                "fold": fold,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "selected_alpha": float(search.best_params_["model__alpha"]),
                "inner_cv_mae": float(-search.best_score_),
            }
        )

    if not np.isfinite(predictions).all() or (fold_ids < 0).any():
        raise RuntimeError("nested CV did not produce one prediction per row")

    final_inner = KFold(
        n_splits=min(inner_splits, len(y)),
        shuffle=True,
        random_state=random_state,
    )
    final_search = GridSearchCV(
        _ridge_pipeline(),
        param_grid={"model__alpha": alpha_grid},
        scoring="neg_mean_absolute_error",
        cv=final_inner,
        n_jobs=n_jobs,
        refit=True,
    )
    final_search.fit(x, y)
    return predictions, fold_ids, fold_details, final_search


def _safe_correlation(function: Any, observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) < 3 or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        return None
    statistic = float(function(observed, predicted).statistic)
    return statistic if np.isfinite(statistic) else None


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Metrics for one set of out-of-fold delta-delta-G predictions."""

    nonzero = observed != 0
    strong = np.abs(observed) >= 1.0
    return {
        "n_rows": int(len(observed)),
        "spearman_r": _safe_correlation(spearmanr, observed, predicted),
        "pearson_r": _safe_correlation(pearsonr, observed, predicted),
        "mae_kcal_mol": float(mean_absolute_error(observed, predicted)),
        "rmse_kcal_mol": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "sign_agreement": (
            float(np.mean(np.sign(observed[nonzero]) == np.sign(predicted[nonzero])))
            if nonzero.any()
            else None
        ),
        "sign_n": int(nonzero.sum()),
        "sign_agreement_abs_ddg_ge_1": (
            float(np.mean(np.sign(observed[strong]) == np.sign(predicted[strong])))
            if strong.any()
            else None
        ),
        "sign_n_abs_ddg_ge_1": int(strong.sum()),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_peptide_dataset(args.modeling_root)
    label_column = LABEL_COLUMNS[args.label]
    systems = sorted(dataset.labels["system"].astype(str).unique())
    if args.systems:
        requested = set(args.systems)
        missing = requested.difference(systems)
        if missing:
            raise ValueError(f"requested systems not found: {sorted(missing)}")
        systems = [system for system in systems if system in requested]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_root = args.out_dir / "models"
    model_root.mkdir(exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_records: dict[str, Any] = {}

    for system_index, system in enumerate(systems):
        rows, blocks, vectorizer = _system_feature_blocks(dataset, system)
        y = pd.to_numeric(rows[label_column], errors="raise").to_numpy(dtype=float)
        system_model_dir = model_root / system
        system_model_dir.mkdir(exist_ok=True)
        fold_records[system] = {}

        for view in args.views:
            x, feature_names = _assemble_view(blocks, view)
            predicted, fold_ids, folds, final_search = _nested_ridge_oof(
                x,
                y,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
                random_state=args.random_state + system_index,
                n_jobs=args.n_jobs,
            )
            metrics = regression_metrics(y, predicted)
            metrics.update(
                {
                    "system": system,
                    "view": view,
                    "model": "ridge",
                    "label": label_column,
                    "split": "random_variant_nested_cv",
                    "n_features": int(x.shape[1]),
                    "outer_splits": int(len(folds)),
                    "final_alpha": float(final_search.best_params_["model__alpha"]),
                }
            )
            metric_rows.append(metrics)
            fold_records[system][view] = folds

            predictions = rows[
                [
                    "system",
                    "input_id",
                    "mutation",
                    "n_substitutions",
                    "n_measurements",
                    label_column,
                ]
            ].copy()
            predictions = predictions.rename(columns={label_column: "observed_ddg_kcal_mol"})
            predictions["predicted_ddg_kcal_mol"] = predicted
            predictions["outer_fold"] = fold_ids
            predictions["view"] = view
            predictions["model"] = "ridge"
            prediction_frames.append(predictions)

            safe_view = view.replace("+", "__")
            joblib.dump(
                {
                    "model": final_search.best_estimator_,
                    "system": system,
                    "view": view,
                    "label_column": label_column,
                    "feature_names": feature_names,
                    "mutation_vocabulary": (
                        vectorizer.vocabulary_ if "mutation" in view.split("+") else None
                    ),
                    "wt_difference_embeddings": True,
                },
                system_model_dir / f"{safe_view}.joblib",
            )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["system", "view"])
    predictions_frame = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["system", "view", "input_id"]
    )
    metrics_frame.to_csv(args.out_dir / "metrics.tsv", sep="\t", index=False)
    predictions_frame.to_csv(args.out_dir / "predictions.tsv", sep="\t", index=False)

    summary = {
        "status": "fit",
        "modeling_root": str(args.modeling_root),
        "out_dir": str(args.out_dir),
        "label_column": label_column,
        "split": "random_variant_nested_cv",
        "systems": systems,
        "views": list(args.views),
        "n_label_rows_with_wt": int(len(dataset.labels)),
        "n_measurement_rows": int(len(dataset.measurements)),
        "metrics": metric_rows,
        "folds": fold_records,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-root", type=Path, default=DEFAULT_MODELING_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--label", choices=sorted(LABEL_COLUMNS), default="median")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run(args)
    headline = pd.DataFrame(summary["metrics"])[
        ["system", "view", "spearman_r", "sign_agreement", "mae_kcal_mol"]
    ]
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
