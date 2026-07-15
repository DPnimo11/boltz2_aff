"""Per-target classification from LRIP interaction-profile features.

Parses the ligand-residue interaction profiles in
``data/ulvsh/modeling/features/lrip/<TARGET>.dat``, joins them to the ULVSH
labels, and fits one Random-Forest classifier per target using the exact same
methodology as the embedding models (``boltz2_aff.modeling.train_classifier``:
median-impute -> standardize -> RF, StratifiedGroupKFold, ``cv_roc_auc``).

For every target it reports, on the *same* LRIP rows:

- ``lrip``   -- RF on the per-residue LRIP features (the new result),
- ``emb``    -- RF on the Boltz affinity embeddings (learned-latent baseline),
- ``B2-A``   -- raw Boltz ``affinity_pred_value``   (lower = stronger binder),
- ``B2-C``   -- raw Boltz ``affinity_probability_binary`` (higher = more active).

Outputs to ``runs/lrip/``:

- ``<TARGET>/lrip/`` and ``<TARGET>/emb/`` -- per-model artifacts from
  ``train_classifier`` (metrics json, predictions, joblib),
- ``summary_by_target.json`` / ``summary.tsv`` -- one row per target,
- ``summary.json`` -- paper-style medians across targets plus run metadata.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from boltz2_aff.features import (
    discover_boltz_scalar_frame,
    discover_embedding_frame,
    feature_columns,
)
from boltz2_aff.modeling import boltz_baseline_metrics, train_classifier

ALL_TARGETS = [
    "ADRA2B", "CASR", "CNR1", "CNR2", "DRD3",
    "DRD4", "MTR1A", "ROCK1", "SC6A4", "SGMR2",
]

REPO = Path(__file__).resolve().parents[1]
LRIP_DIR = REPO / "data/ulvsh/modeling/features/lrip"
LABELS = REPO / "data/ulvsh/modeling/labels.tsv"
BOLTZ_SCALARS = REPO / "data/ulvsh/modeling/features/boltz_scalars.tsv"
EMBEDDING_ROOT = REPO / "data/ulvsh/modeling/features/boltz_embeddings"
OUT_ROOT = REPO / "runs/lrip"

# Known one-off id normalizations from the .dat writer back to labels.tsv ids.
LIGAND_ID_FIXES: dict[str, dict[str, str]] = {
    "DRD3": {"1_20": "1_2_0"},
}

B2A = "boltz_affinity_pred_value"          # lower = stronger binder
B2C = "boltz_affinity_probability_binary"  # higher = more active


def parse_lrip(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame with ligand_id + lrip_* columns, list of lrip feature names)."""
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
    resids = header[1:]
    # Position-tagged so duplicated residue numbers stay distinct and stable.
    names = [f"lrip_{i:03d}_r{resid}" for i, resid in enumerate(resids)]
    frame = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, dtype={0: str})
    if frame.shape[1] != len(names) + 1:
        raise ValueError(
            f"{path.name}: {frame.shape[1] - 1} value columns but {len(names)} header residues"
        )
    frame.columns = ["ligand_id", *names]
    return frame, names


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABELS, sep="\t")
    labels["variant"] = "wt"
    return labels


def build_target_frame(
    target: str,
    labels: pd.DataFrame,
    boltz: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    lrip, lrip_cols = parse_lrip(LRIP_DIR / f"{target}.dat")
    fixes = LIGAND_ID_FIXES.get(target, {})
    if fixes:
        lrip["ligand_id"] = lrip["ligand_id"].replace(fixes)
    lrip["target"] = target

    lab_t = labels[labels["target"] == target]
    frame = lab_t.merge(lrip, on=["target", "ligand_id"], how="inner")

    unmatched = len(lrip) - len(frame)
    if unmatched:
        missing = sorted(set(lrip["ligand_id"]) - set(frame["ligand_id"]))
        raise ValueError(f"{target}: {unmatched} LRIP rows did not match labels: {missing}")

    scalar_cols = [c for c in boltz.columns if c.startswith("boltz_")]
    frame = frame.merge(
        boltz[["target", "variant", "ligand_id", *scalar_cols]],
        on=["target", "variant", "ligand_id"],
        how="left",
    )
    frame["group_id"] = frame["target"].astype(str) + "::" + frame["ligand_id"].astype(str)
    return frame, lrip_cols


def classify(frame: pd.DataFrame, cols: list[str], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return train_classifier(frame, cols, out_dir, max_splits=5, random_state=42)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    boltz = discover_boltz_scalar_frame(BOLTZ_SCALARS, variants=["wt"])
    embeddings = discover_embedding_frame([EMBEDDING_ROOT], variants=["wt"])
    emb_cols_all = [c for c in embeddings.columns if c.startswith("emb_")]

    rows: list[dict[str, Any]] = []
    for target in ALL_TARGETS:
        frame, lrip_cols = build_target_frame(target, labels, boltz)
        n = int(len(frame))
        n_pos = int(frame["active_bool"].sum())
        n_neg = n - n_pos

        lrip_metrics = classify(frame, lrip_cols, OUT_ROOT / target / "lrip")

        # Embedding model on the SAME compounds (inner join keeps only shared rows).
        emb_frame = frame.merge(
            embeddings[["target", "ligand_id", *emb_cols_all]],
            on=["target", "ligand_id"],
            how="inner",
        )
        emb_cols = feature_columns(emb_frame, "embeddings")
        if emb_cols and len(emb_frame) > 0:
            emb_metrics = classify(emb_frame, emb_cols, OUT_ROOT / target / "emb")
        else:
            emb_metrics = {"status": "skipped", "reason": "no embeddings for these rows"}

        baseline = boltz_baseline_metrics(frame).get("per_target", {}).get(target, {})

        row = {
            "target": target,
            "n_rows": n,
            "n_active": n_pos,
            "n_inactive": n_neg,
            "n_lrip_features": len(lrip_cols),
            "cv_splits": lrip_metrics.get("cv_splits"),
            "lrip_cv_roc_auc": lrip_metrics.get("cv_roc_auc"),
            "lrip_cv_average_precision": lrip_metrics.get("cv_average_precision"),
            "emb_n_rows": int(len(emb_frame)),
            "emb_cv_roc_auc": emb_metrics.get("cv_roc_auc"),
            "b2a_roc_auc": baseline.get(f"{B2A}_roc_auc"),
            "b2c_roc_auc": baseline.get(f"{B2C}_roc_auc"),
            "lrip_status": lrip_metrics.get("status"),
            "lrip_note": lrip_metrics.get("cv_note") or lrip_metrics.get("reason"),
        }
        # Best-of raw Boltz, the paper's per-target reference point.
        raw = [v for v in (row["b2a_roc_auc"], row["b2c_roc_auc"]) if v is not None]
        row["raw_boltz_best"] = max(raw) if raw else None
        rows.append(row)
        print(
            f"{target:7s} n={n:3d} (+{n_pos:3d}/-{n_neg:3d})  "
            f"LRIP={_fmt(row['lrip_cv_roc_auc'])}  emb={_fmt(row['emb_cv_roc_auc'])}  "
            f"B2-A={_fmt(row['b2a_roc_auc'])}  B2-C={_fmt(row['b2c_roc_auc'])}"
            + ("" if row["cv_splits"] else f"   [{row['lrip_note']}]")
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT_ROOT / "summary.tsv", sep="\t", index=False)
    (OUT_ROOT / "summary_by_target.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    medians = {
        key: _median([r[key] for r in rows])
        for key in ("lrip_cv_roc_auc", "emb_cv_roc_auc", "b2a_roc_auc", "b2c_roc_auc", "raw_boltz_best")
    }
    # Head-to-head only over targets where LRIP produced a CV score.
    scored = [r for r in rows if r["lrip_cv_roc_auc"] is not None]
    summary = {
        "n_targets": len(rows),
        "n_targets_scored": len(scored),
        "targets_skipped": [r["target"] for r in rows if r["lrip_cv_roc_auc"] is None],
        "median_across_targets": medians,
        "lrip_beats_raw_boltz_best": sum(
            1 for r in scored
            if r["raw_boltz_best"] is not None and r["lrip_cv_roc_auc"] > r["raw_boltz_best"]
        ),
        "lrip_beats_embeddings": sum(
            1 for r in scored
            if r["emb_cv_roc_auc"] is not None and r["lrip_cv_roc_auc"] > r["emb_cv_roc_auc"]
        ),
        "lrip_acceptable_auc_gt_0_65": sum(
            1 for r in scored if r["lrip_cv_roc_auc"] > 0.65
        ),
        "methodology": {
            "classifier": "RandomForestClassifier(n_estimators=200, class_weight=balanced, max_features=sqrt)",
            "pipeline": "median impute -> StandardScaler -> RF",
            "cv": "StratifiedGroupKFold, group=ligand, up to 5 splits",
            "metric": "cross_val_predict proba -> roc_auc_score",
            "reused_from": "boltz2_aff.modeling.train_classifier / boltz_baseline_metrics",
        },
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n== medians across targets ==")
    for key, value in medians.items():
        print(f"  {key:20s} {_fmt(value)}")
    print(
        f"\nLRIP scored on {len(scored)}/{len(rows)} targets; "
        f"beats best raw Boltz on {summary['lrip_beats_raw_boltz_best']}/{len(scored)}, "
        f"beats embeddings on {summary['lrip_beats_embeddings']}/{len(scored)}, "
        f"AUC>0.65 on {summary['lrip_acceptable_auc_gt_0_65']}/{len(scored)}."
    )
    print(f"\nwrote {OUT_ROOT}/summary.tsv, summary_by_target.json, summary.json")


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) and value is not None else "  -  "


def _median(values: list[Any]) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float)) and v is not None]
    return float(statistics.median(clean)) if clean else None


if __name__ == "__main__":
    main()
