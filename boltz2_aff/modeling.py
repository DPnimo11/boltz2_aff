"""Model fitting and evaluation for affinity feature tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _as_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    return frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def _classification_cv(y: np.ndarray, groups: np.ndarray, max_splits: int, random_state: int):
    group_labels = pd.DataFrame({"group": groups, "y": y}).drop_duplicates("group")
    class_counts = group_labels["y"].value_counts()
    if len(class_counts) < 2:
        return None
    n_splits = min(max_splits, int(class_counts.min()), group_labels["group"].nunique())
    if n_splits < 2:
        return None
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _regression_cv(groups: np.ndarray, max_splits: int):
    n_splits = min(max_splits, len(set(groups)))
    if n_splits < 2:
        return None
    return GroupKFold(n_splits=n_splits)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_float):
        return None
    return value_float


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    serializable = {
        key: _safe_float(value) if isinstance(value, (np.floating, float)) else value
        for key, value in payload.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def train_classifier(
    frame: pd.DataFrame,
    feature_columns: list[str],
    out_dir: Path,
    max_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    rows = frame[frame["active_bool"].notna()].copy()
    y = rows["active_bool"].astype(bool).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return {"status": "skipped", "reason": "classification requires both active classes"}

    x = _as_feature_matrix(rows, feature_columns)
    groups = rows["group_id"].astype(str).to_numpy()
    model = Pipeline(
        [
            ("imputer", _imputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )

    metrics: dict[str, Any] = {
        "status": "fit",
        "n_rows": int(len(rows)),
        "n_features": int(len(feature_columns)),
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
    }
    cv = _classification_cv(y, groups, max_splits, random_state)
    if cv is not None:
        proba = cross_val_predict(model, x, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics.update(
            {
                "cv_splits": int(cv.n_splits),
                "cv_accuracy": accuracy_score(y, pred),
                "cv_balanced_accuracy": balanced_accuracy_score(y, pred),
                "cv_roc_auc": roc_auc_score(y, proba),
                "cv_average_precision": average_precision_score(y, proba),
                "cv_log_loss": log_loss(y, proba, labels=[0, 1]),
            }
        )
        predictions = rows[["target", "variant", "ligand_id", "active_bool", "group_id"]].copy()
        predictions["pred_active_probability"] = proba
        predictions["pred_active_bool"] = pred.astype(bool)
        predictions.to_csv(out_dir / "predictions_classification.csv", index=False)
    else:
        metrics["cv_splits"] = 0
        metrics["cv_note"] = "not enough grouped class balance for cross-validation"

    final_model = clone(model).fit(x, y)
    joblib.dump({"model": final_model, "feature_columns": feature_columns}, out_dir / "classifier.joblib")
    _write_json(out_dir / "metrics_classification.json", metrics)
    return metrics


def train_regressor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    out_dir: Path,
    max_splits: int = 5,
) -> dict[str, Any]:
    rows = frame[frame["p_affinity"].notna()].copy()
    if len(rows) < 3:
        return {"status": "skipped", "reason": "regression requires at least three numeric affinity rows"}

    y = rows["p_affinity"].astype(float).to_numpy()
    x = _as_feature_matrix(rows, feature_columns)
    groups = rows["group_id"].astype(str).to_numpy()
    model = Pipeline(
        [
            ("imputer", _imputer()),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 33))),
        ]
    )

    metrics: dict[str, Any] = {
        "status": "fit",
        "n_rows": int(len(rows)),
        "n_features": int(len(feature_columns)),
    }
    cv = _regression_cv(groups, max_splits)
    if cv is not None:
        pred = cross_val_predict(model, x, y, groups=groups, cv=cv)
        metrics.update(
            {
                "cv_splits": int(cv.n_splits),
                "cv_rmse_p_affinity": float(np.sqrt(mean_squared_error(y, pred))),
                "cv_mae_p_affinity": mean_absolute_error(y, pred),
                "cv_r2": r2_score(y, pred),
                "cv_pearson_r": pearsonr(y, pred).statistic,
                "cv_spearman_r": spearmanr(y, pred).statistic,
            }
        )
        predictions = rows[
            ["target", "variant", "ligand_id", "affinity_source", "affinity_um", "p_affinity", "group_id"]
        ].copy()
        predictions["pred_p_affinity"] = pred
        predictions["pred_affinity_um"] = np.power(10.0, 6.0 - pred)
        predictions.to_csv(out_dir / "predictions_regression.csv", index=False)
    else:
        metrics["cv_splits"] = 0
        metrics["cv_note"] = "not enough groups for cross-validation"

    final_model = clone(model).fit(x, y)
    joblib.dump({"model": final_model, "feature_columns": feature_columns}, out_dir / "regressor.joblib")
    _write_json(out_dir / "metrics_regression.json", metrics)
    return metrics
