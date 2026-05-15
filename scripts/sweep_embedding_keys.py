"""Sweep Boltz affinity embedding components and summarize ROC AUC / regression metrics.

Runs the pipeline once per combination listed in COMBOS, restricted to a single target,
writing each result into runs/<target>_sweep/<combo_name>/ and emitting a summary table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

COMBOS: list[tuple[str, list[str]]] = [
    ("pair_mean1", ["pair_mean1"]),
    ("pair_mean2", ["pair_mean2"]),
    ("head1", ["head1"]),
    ("head2", ["head2"]),
    ("pair_mean1_pair_mean2", ["pair_mean1", "pair_mean2"]),
    ("head1_head2", ["head1", "head2"]),
    ("pair_mean1_head1", ["pair_mean1", "head1"]),
    ("pair_mean2_head2", ["pair_mean2", "head2"]),
    ("all", ["pair_mean1", "head1", "pair_mean2", "head2"]),
]

METRIC_KEYS = [
    ("classification", "cv_roc_auc"),
    ("classification", "cv_average_precision"),
    ("regression", "cv_roc_auc"),
    ("regression", "cv_pearson_r"),
    ("regression", "cv_spearman_r"),
    ("regression", "cv_r2"),
    ("regression", "cv_rmse_p_affinity"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="ROCK1")
    parser.add_argument("--out-root", type=Path, default=Path("runs/rock1_sweep"))
    parser.add_argument("--feature-set", default="embeddings", choices=["embeddings", "combined"])
    parser.add_argument(
        "--combos",
        nargs="*",
        default=None,
        help="Optional subset of combo names. Defaults to all.",
    )
    return parser.parse_args()


def _run_pipeline(combo_name: str, keys: list[str], target: str, out_dir: Path, feature_set: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "boltz2_aff.pipeline",
        "--targets",
        target,
        "--feature-set",
        feature_set,
        "--embedding-keys",
        *keys,
        "--out-dir",
        str(out_dir),
    ]
    print(f"[sweep] {combo_name}: running {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest


def _summarize(manifest: dict, combo_name: str, keys: list[str]) -> dict:
    metrics = manifest.get("metrics", {})
    summary = {
        "combo": combo_name,
        "keys": "+".join(keys),
        "n_features": manifest.get("n_features"),
    }
    for section, key in METRIC_KEYS:
        value = metrics.get(section, {}).get(key)
        summary[f"{section}.{key}"] = round(value, 4) if isinstance(value, (int, float)) else None
    baseline = metrics.get("boltz_baseline", {}).get("per_target", {}).get(manifest.get("targets_requested", [None])[0], {})
    summary["baseline_B2C_roc_auc"] = round(baseline.get("boltz_affinity_probability_binary_roc_auc", float("nan")), 4) if "boltz_affinity_probability_binary_roc_auc" in baseline else None
    summary["baseline_B2A_roc_auc"] = round(baseline.get("boltz_affinity_pred_value_roc_auc", float("nan")), 4) if "boltz_affinity_pred_value_roc_auc" in baseline else None
    return summary


def main() -> None:
    args = _parse_args()
    selected = COMBOS
    if args.combos:
        wanted = set(args.combos)
        selected = [c for c in COMBOS if c[0] in wanted]
        if not selected:
            raise SystemExit(f"no combos matched {args.combos}")

    rows: list[dict] = []
    for combo_name, keys in selected:
        out_dir = args.out_root / combo_name
        manifest = _run_pipeline(combo_name, keys, args.target, out_dir, args.feature_set)
        rows.append(_summarize(manifest, combo_name, keys))

    summary_path = args.out_root / "summary.json"
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    columns = list(rows[0].keys())
    widths = {col: max(len(col), *(len(str(r[col])) for r in rows)) for col in columns}
    line = " | ".join(col.ljust(widths[col]) for col in columns)
    print()
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in columns))
    print(f"\n[sweep] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
