"""Nested cross-validation for unbiased evaluation of embedding combo selection.

For each target, outer CV folds evaluate performance after inner-fold combo
selection. Reports nested-CV AUC alongside fixed pair_mean1 AUC computed on
the same outer folds (exact apples-to-apples) and the raw B2-C baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from boltz2_aff.data import load_ulvsh
from boltz2_aff.features import (
    discover_boltz_scalar_frame,
    discover_embedding_frame,
    feature_columns,
)
from boltz2_aff.modeling import _as_feature_matrix, _build_classifier_pipeline

ALL_TARGETS = [
    "ADRA2B", "CASR", "CNR1", "CNR2", "DRD3",
    "DRD4", "MTR1A", "ROCK1", "SC6A4", "SGMR2",
]

COMBOS: dict[str, list[str]] = {
    "pair_mean1": ["pair_mean1"],
    "pair_mean2": ["pair_mean2"],
    "head1": ["head1"],
    "head2": ["head2"],
    "pair_mean1+pair_mean2": ["pair_mean1", "pair_mean2"],
    "head1+head2": ["head1", "head2"],
    "pair_mean1+head1": ["pair_mean1", "head1"],
    "pair_mean2+head2": ["pair_mean2", "head2"],
    "all": ["pair_mean1", "head1", "pair_mean2", "head2"],
}


def _col_mask(all_cols: list[str], emb_keys: list[str]) -> np.ndarray:
    """Boolean mask over all_cols: keep non-emb cols and emb cols matching any key."""
    mask = np.zeros(len(all_cols), dtype=bool)
    for i, col in enumerate(all_cols):
        if not col.startswith("emb_"):
            mask[i] = True
        elif any(k in col for k in emb_keys):
            mask[i] = True
    return mask


def _b2c_auc(frame: pd.DataFrame) -> float | None:
    col = "boltz_affinity_probability_binary"
    if col not in frame.columns or "active_bool" not in frame.columns:
        return None
    rows = frame[frame["active_bool"].notna()]
    y = rows["active_bool"].astype(bool).astype(int).to_numpy()
    scores = pd.to_numeric(rows[col], errors="coerce").to_numpy()
    valid = np.isfinite(scores)
    if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
        return None
    return round(float(roc_auc_score(y[valid], scores[valid])), 4)


def nested_cv_target(
    frame: pd.DataFrame,
    feat_cols: list[str],
    outer_splits: int = 3,
    inner_splits: int = 3,
    random_state: int = 42,
) -> dict:
    rows = frame[frame["active_bool"].notna()].copy().reset_index(drop=True)
    y = rows["active_bool"].astype(bool).astype(int).to_numpy()
    groups = rows["group_id"].astype(str).to_numpy()

    if len(np.unique(y)) < 2:
        return {"status": "skipped", "reason": "need both classes"}

    n_outer = min(outer_splits, int(np.bincount(y).min()), len(set(groups)))
    if n_outer < 2:
        return {"status": "skipped", "reason": "too few samples for outer CV"}

    outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    x_full = _as_feature_matrix(rows, feat_cols)
    masks = {name: _col_mask(feat_cols, keys) for name, keys in COMBOS.items()}

    nested_aucs: list[float] = []
    fixed_pm1_aucs: list[float] = []
    selected_combos: list[str] = []
    fold_details: list[dict] = []

    for train_idx, test_idx in outer_cv.split(x_full, y, groups=groups):
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Fixed pair_mean1 on this outer fold (reference)
        m_pm1 = masks["pair_mean1"]
        model = _build_classifier_pipeline(random_state)
        model.fit(x_full[np.ix_(train_idx, m_pm1)], y_train)
        proba = model.predict_proba(x_full[np.ix_(test_idx, m_pm1)])[:, 1]
        fixed_pm1_aucs.append(float(roc_auc_score(y_test, proba)))

        # Inner CV: select best combo on training split
        n_inner = min(inner_splits, int(np.bincount(y_train).min()), len(set(g_train)))
        inner_scores: dict[str, float] = {}

        if n_inner >= 2:
            inner_cv = StratifiedGroupKFold(
                n_splits=n_inner, shuffle=True, random_state=random_state
            )
            for combo_name, m_combo in masks.items():
                x_train = x_full[np.ix_(train_idx, m_combo)]
                try:
                    scores = cross_val_score(
                        _build_classifier_pipeline(random_state),
                        x_train, y_train,
                        groups=g_train, cv=inner_cv, scoring="roc_auc",
                    )
                    inner_scores[combo_name] = round(float(np.mean(scores)), 4)
                except Exception:
                    inner_scores[combo_name] = float("nan")
            valid = {k: v for k, v in inner_scores.items() if np.isfinite(v)}
            best_combo = max(valid, key=valid.__getitem__) if valid else "pair_mean1"
        else:
            best_combo = "pair_mean1"

        selected_combos.append(best_combo)

        # Outer evaluation with the inner-selected combo
        m_best = masks[best_combo]
        model = _build_classifier_pipeline(random_state)
        model.fit(x_full[np.ix_(train_idx, m_best)], y_train)
        proba = model.predict_proba(x_full[np.ix_(test_idx, m_best)])[:, 1]
        nested_aucs.append(float(roc_auc_score(y_test, proba)))

        fold_details.append({
            "best_combo": best_combo,
            "inner_scores": inner_scores,
            "nested_outer_auc": round(nested_aucs[-1], 4),
            "fixed_pm1_outer_auc": round(fixed_pm1_aucs[-1], 4),
        })

    if not nested_aucs:
        return {"status": "skipped", "reason": "no valid outer folds"}

    return {
        "status": "fit",
        "n_rows": int(len(rows)),
        "positive_rows": int(y.sum()),
        "negative_rows": int((1 - y).sum()),
        "outer_splits_used": len(nested_aucs),
        "nested_cv_mean_auc": round(float(np.mean(nested_aucs)), 4),
        "fixed_pm1_mean_auc": round(float(np.mean(fixed_pm1_aucs)), 4),
        "fold_details": fold_details,
        "selected_combos": selected_combos,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=ALL_TARGETS)
    p.add_argument("--ulvsh-root", type=Path, default=Path("data/ulvsh/source"))
    p.add_argument(
        "--boltz-scalar-source",
        "--boltz-output-root",
        dest="boltz_scalar_source",
        type=Path,
        default=Path("data/ulvsh/modeling/features/boltz_scalars.tsv"),
    )
    p.add_argument(
        "--embedding-root",
        type=Path,
        action="append",
        default=[Path("data/ulvsh/modeling/features/boltz_embeddings")],
        help="Root(s) containing affinity_embeddings_*.npz files.",
    )
    p.add_argument("--outer-splits", type=int, default=3)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("runs/nested_cv.json"))
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    embedding_roots = list(dict.fromkeys(args.embedding_root))

    all_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    print(f"{'Target':<10} {'nested':>8} {'fixed_pm1':>10} {'B2C':>8}  {'n':>4}  selected")
    print("-" * 72)

    for target in args.targets:
        labels = load_ulvsh(args.ulvsh_root, [target], "raw", include_scores=True)
        embeddings = discover_embedding_frame(embedding_roots, [target])
        boltz_scalars = discover_boltz_scalar_frame(args.boltz_scalar_source, [target])

        if embeddings.empty:
            print(f"{target:<10}  no embeddings")
            all_results[target] = {"status": "no_embeddings"}
            continue

        frame = labels.merge(embeddings, on=["target", "ligand_id"], how="inner")
        if not boltz_scalars.empty:
            scalar_cols = [c for c in boltz_scalars.columns if c.startswith("boltz_")]
            frame = frame.merge(
                boltz_scalars[["target", "variant", "ligand_id", *scalar_cols]],
                on=["target", "variant", "ligand_id"],
                how="left",
            )
        frame["group_id"] = frame["target"].astype(str) + "::" + frame["ligand_id"].astype(str)
        feat_cols = feature_columns(frame, "combined")

        b2c = _b2c_auc(frame)
        result = nested_cv_target(
            frame, feat_cols,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            random_state=args.random_state,
        )
        result["b2c_auc"] = b2c
        all_results[target] = result

        if result["status"] != "fit":
            print(f"{target:<10}  {result.get('reason', 'skipped')}")
            continue

        nested = result["nested_cv_mean_auc"]
        pm1 = result["fixed_pm1_mean_auc"]
        n = result["n_rows"]
        b2c_str = f"{b2c:.4f}" if b2c is not None else "   N/A"
        combo_str = ", ".join(result["selected_combos"])
        print(f"{target:<10} {nested:>8.4f} {pm1:>10.4f} {b2c_str:>8}  {n:>4}  [{combo_str}]")

        summary_rows.append({"target": target, "nested": nested, "fixed_pm1": pm1, "b2c": b2c, "n": n})

    if len(summary_rows) >= 2:
        def _med(vals: list) -> float:
            s = sorted(v for v in vals if v is not None)
            return s[len(s) // 2] if s else float("nan")

        med_n = _med([r["nested"] for r in summary_rows])
        med_pm1 = _med([r["fixed_pm1"] for r in summary_rows])
        med_b2c = _med([r["b2c"] for r in summary_rows])
        print("-" * 72)
        print(f"{'Median':<10} {med_n:>8.4f} {med_pm1:>10.4f} {med_b2c:>8.4f}")

        n_nested_wins = sum(
            1 for r in summary_rows
            if r["nested"] is not None and r["b2c"] is not None and r["nested"] > r["b2c"]
        )
        n_pm1_wins = sum(
            1 for r in summary_rows
            if r["fixed_pm1"] is not None and r["b2c"] is not None and r["fixed_pm1"] > r["b2c"]
        )
        print(f"\nnested > B2C: {n_nested_wins}/{len(summary_rows)}   fixed_pm1 > B2C: {n_pm1_wins}/{len(summary_rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
