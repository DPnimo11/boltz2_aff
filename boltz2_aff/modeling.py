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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
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


BOLTZ_BASELINE_SCORE_COLUMNS: tuple[str, ...] = (
    "boltz_affinity_pred_value",
    "boltz_affinity_probability_binary",
)


def _is_higher_score_more_active(column: str) -> bool:
    """Return True when larger raw values should rank actives above inactives."""
    return column != "boltz_affinity_pred_value"


def _auc_against_active(scores: np.ndarray, labels: np.ndarray, higher_is_active: bool) -> float | None:
    if scores.size == 0 or len(np.unique(labels)) < 2:
        return None
    oriented = scores if higher_is_active else -scores
    return float(roc_auc_score(labels, oriented))


def boltz_baseline_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Per-target ROC AUC and rank correlations from raw Boltz scalar outputs."""

    available_scores = [
        column for column in BOLTZ_BASELINE_SCORE_COLUMNS if column in frame.columns
    ]
    if not available_scores:
        return {"status": "skipped", "reason": "no Boltz scalar columns in dataset"}

    targets = sorted(frame["target"].dropna().unique().tolist())
    per_target: dict[str, dict[str, Any]] = {}
    for target in targets:
        rows = frame[frame["target"] == target]
        target_metrics: dict[str, Any] = {
            "n_rows": int(len(rows)),
            "n_with_active": int(rows["active_bool"].notna().sum()) if "active_bool" in rows else 0,
            "n_with_p_affinity": int(rows["p_affinity"].notna().sum()) if "p_affinity" in rows else 0,
        }
        if "active_bool" in rows.columns:
            active_mask = rows["active_bool"].notna().to_numpy()
            active_labels = rows.loc[active_mask, "active_bool"].astype(bool).astype(int).to_numpy()
            for column in available_scores:
                scores = pd.to_numeric(rows.loc[active_mask, column], errors="coerce")
                valid = scores.notna().to_numpy()
                auc = _auc_against_active(
                    scores.to_numpy()[valid],
                    active_labels[valid],
                    higher_is_active=_is_higher_score_more_active(column),
                )
                if auc is not None:
                    target_metrics[f"{column}_roc_auc"] = auc
                    target_metrics[f"{column}_n_for_auc"] = int(valid.sum())
        if "p_affinity" in rows.columns:
            regression_rows = rows[rows["p_affinity"].notna()]
            y = pd.to_numeric(regression_rows["p_affinity"], errors="coerce").to_numpy()
            for column in available_scores:
                preds = pd.to_numeric(regression_rows[column], errors="coerce").to_numpy()
                mask = np.isfinite(preds) & np.isfinite(y)
                if mask.sum() >= 3:
                    sign = 1.0 if _is_higher_score_more_active(column) else -1.0
                    target_metrics[f"{column}_pearson_r"] = float(pearsonr(y[mask], sign * preds[mask]).statistic)
                    target_metrics[f"{column}_spearman_r"] = float(spearmanr(y[mask], sign * preds[mask]).statistic)
                    target_metrics[f"{column}_n_for_corr"] = int(mask.sum())
        per_target[target] = target_metrics

    summary: dict[str, Any] = {"status": "fit", "score_columns": list(available_scores), "per_target": per_target}
    for column in available_scores:
        aucs = [m.get(f"{column}_roc_auc") for m in per_target.values() if m.get(f"{column}_roc_auc") is not None]
        if aucs:
            summary[f"{column}_median_roc_auc"] = float(np.median(aucs))
            summary[f"{column}_mean_roc_auc"] = float(np.mean(aucs))
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    serializable = {
        key: _safe_float(value) if isinstance(value, (np.floating, float)) else value
        for key, value in payload.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")


def _build_classifier_pipeline(random_state: int) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()), ("model", classifier)])


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
    model = _build_classifier_pipeline(random_state)

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


def _screening_auc(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model: Pipeline,
    max_splits: int,
) -> dict[str, Any] | None:
    """Score every active-labeled row by training the regressor on numeric-affinity rows in each fold."""

    if "active_bool" not in frame.columns:
        return None
    labeled = frame[frame["active_bool"].notna()].copy()
    if labeled.empty:
        return None
    y_active = labeled["active_bool"].astype(bool).astype(int).to_numpy()
    if len(np.unique(y_active)) < 2:
        return {"cv_roc_auc_note": "need both active classes in dataset"}

    has_target = labeled["p_affinity"].notna().to_numpy()
    if has_target.sum() < 3:
        return {"cv_roc_auc_note": "need at least three numeric affinity rows for screening AUC"}

    groups = labeled["group_id"].astype(str).to_numpy()
    cv = _classification_cv(y_active, groups, max_splits, random_state=42)
    if cv is None:
        return {"cv_roc_auc_note": "not enough grouped class balance for screening AUC CV"}

    x_all = _as_feature_matrix(labeled, feature_columns)
    y_target = pd.to_numeric(labeled["p_affinity"], errors="coerce").to_numpy(dtype=float)
    pred = np.full(len(labeled), np.nan)
    for train_idx, test_idx in cv.split(x_all, y_active, groups=groups):
        train_mask = has_target[train_idx]
        if train_mask.sum() < 2:
            continue
        fitted = clone(model).fit(x_all[train_idx][train_mask], y_target[train_idx][train_mask])
        pred[test_idx] = fitted.predict(x_all[test_idx])

    scored = np.isfinite(pred)
    if scored.sum() == 0 or len(np.unique(y_active[scored])) < 2:
        return {"cv_roc_auc_note": "no usable screening predictions"}
    return {
        "cv_roc_auc": float(roc_auc_score(y_active[scored], pred[scored])),
        "cv_roc_auc_n_rows": int(scored.sum()),
        "cv_roc_auc_positive_rows": int(y_active[scored].sum()),
        "cv_roc_auc_negative_rows": int((1 - y_active[scored]).sum()),
    }


def train_regressor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    out_dir: Path,
    max_splits: int = 5,
    boltz_residual_column: str | None = "boltz_affinity_pred_value",
) -> dict[str, Any]:
    rows = frame[frame["p_affinity"].notna()].copy()
    if len(rows) < 3:
        return {"status": "skipped", "reason": "regression requires at least three numeric affinity rows"}

    # Residual mode: train on p_affinity - boltz_scalar so the model corrects Boltz-2
    boltz_oriented_arr: np.ndarray | None = None
    residual_col = None
    if boltz_residual_column and boltz_residual_column in rows.columns:
        boltz_vals = pd.to_numeric(rows[boltz_residual_column], errors="coerce")
        # boltz_affinity_pred_value is negative (lower = tighter); negate so larger = stronger
        boltz_oriented = -boltz_vals if boltz_residual_column == "boltz_affinity_pred_value" else boltz_vals
        valid_boltz = boltz_oriented.notna()
        if valid_boltz.sum() >= len(rows) * 0.8:
            rows = rows[valid_boltz].copy()
            boltz_oriented_valid = boltz_oriented[valid_boltz]
            boltz_oriented_arr = boltz_oriented_valid.to_numpy(dtype=float)
            rows["_residual"] = rows["p_affinity"].astype(float).values - boltz_oriented_arr
            residual_col = "_residual"

    y_raw = rows["p_affinity"].astype(float).to_numpy()
    y = rows[residual_col].astype(float).to_numpy() if residual_col else y_raw
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
        "residual_mode": residual_col is not None,
        "residual_boltz_column": boltz_residual_column if residual_col else None,
    }
    cv = _regression_cv(groups, max_splits)
    if cv is not None:
        pred_residual = cross_val_predict(model, x, y, groups=groups, cv=cv)
        # Reconstruct absolute p_affinity predictions for residual mode
        if residual_col and boltz_oriented_arr is not None:
            pred = pred_residual + boltz_oriented_arr
        else:
            pred = pred_residual
        metrics.update(
            {
                "cv_splits": int(cv.n_splits),
                "cv_rmse_p_affinity": float(np.sqrt(mean_squared_error(y_raw, pred))),
                "cv_mae_p_affinity": mean_absolute_error(y_raw, pred),
                "cv_r2": r2_score(y_raw, pred),
                "cv_pearson_r": pearsonr(y_raw, pred).statistic,
                "cv_spearman_r": spearmanr(y_raw, pred).statistic,
            }
        )
        screening = _screening_auc(frame, feature_columns, model, max_splits)
        if screening is not None:
            metrics.update(screening)
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
