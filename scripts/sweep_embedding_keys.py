"""Sweep Boltz affinity embedding components per target and summarize metrics.

For every target and every embedding-component combination in COMBOS, runs the
pipeline (one model per target) into <out-root>/<target>/<combo>/, then writes:

- summary.json            flat list of every (target, combo) row
- summary_by_target.json  rows grouped by target
- medians.json            median across targets per combo (paper-style)

and prints, per target, the best combo for each headline metric next to the
raw Boltz-2 B2-A / B2-C baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ALL_TARGETS = [
    "ADRA2B", "CASR", "CNR1", "CNR2", "DRD3",
    "DRD4", "MTR1A", "ROCK1", "SC6A4", "SGMR2",
]

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

HEADLINE = [
    ("classification.cv_roc_auc", True),
    ("regression.cv_pearson_r", True),
    ("regression.cv_spearman_r", True),
    ("regression.cv_r2", True),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=ALL_TARGETS)
    parser.add_argument("--out-root", type=Path, default=Path("runs/sweep"))
    parser.add_argument("--feature-set", default="embeddings", choices=["embeddings", "combined"])
    parser.add_argument("--combos", nargs="*", default=None, help="Optional subset of combo names.")
    return parser.parse_args()


def _run_pipeline(keys: list[str], target: str, out_dir: Path, feature_set: str) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "boltz2_aff.pipeline",
        "--targets", target,
        "--feature-set", feature_set,
        "--embedding-keys", *keys,
        "--out-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[sweep] {target}/{'+'.join(keys)} FAILED: {result.stderr.strip().splitlines()[-1:]}")
        return None
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def _summarize(manifest: dict, target: str, combo_name: str, keys: list[str]) -> dict:
    metrics = manifest.get("metrics", {})
    summary: dict = {
        "target": target,
        "combo": combo_name,
        "keys": "+".join(keys),
        "n_features": manifest.get("n_features"),
    }
    for section, key in METRIC_KEYS:
        value = metrics.get(section, {}).get(key)
        summary[f"{section}.{key}"] = round(value, 4) if isinstance(value, (int, float)) else None
    baseline = metrics.get("boltz_baseline", {}).get("per_target", {}).get(target, {})
    for tag, col in [("B2C", "boltz_affinity_probability_binary_roc_auc"),
                     ("B2A", "boltz_affinity_pred_value_roc_auc")]:
        summary[f"baseline_{tag}_roc_auc"] = round(baseline[col], 4) if col in baseline else None
    return summary


def _best(rows: list[dict], metric: str, higher_is_better: bool) -> dict | None:
    valid = [r for r in rows if isinstance(r.get(metric), (int, float))]
    if not valid:
        return None
    return (max if higher_is_better else min)(valid, key=lambda r: r[metric])


def main() -> None:
    args = _parse_args()
    selected = COMBOS
    if args.combos:
        wanted = set(args.combos)
        selected = [c for c in COMBOS if c[0] in wanted]
        if not selected:
            raise SystemExit(f"no combos matched {args.combos}")

    all_rows: list[dict] = []
    by_target: dict[str, list[dict]] = {}
    for target in args.targets:
        target_rows: list[dict] = []
        for combo_name, keys in selected:
            out_dir = args.out_root / target / combo_name
            manifest = _run_pipeline(keys, target, out_dir, args.feature_set)
            if manifest is None:
                continue
            row = _summarize(manifest, target, combo_name, keys)
            target_rows.append(row)
            all_rows.append(row)
            print(f"[sweep] {target:<7} {combo_name:<22} "
                  f"clsAUC={row.get('classification.cv_roc_auc')} "
                  f"regP={row.get('regression.cv_pearson_r')} "
                  f"B2C={row.get('baseline_B2C_roc_auc')}")
        by_target[target] = target_rows

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (args.out_root / "summary_by_target.json").write_text(json.dumps(by_target, indent=2), encoding="utf-8")

    medians: dict[str, dict] = {}
    for combo_name, _ in selected:
        combo_rows = [r for r in all_rows if r["combo"] == combo_name]
        entry: dict = {"n_targets": len(combo_rows)}
        for section, key in METRIC_KEYS:
            vals = [r[f"{section}.{key}"] for r in combo_rows if isinstance(r.get(f"{section}.{key}"), (int, float))]
            if vals:
                entry[f"median_{section}.{key}"] = round(statistics.median(vals), 4)
        medians[combo_name] = entry
    (args.out_root / "medians.json").write_text(json.dumps(medians, indent=2), encoding="utf-8")

    print("\n=== Per-target best combo vs raw Boltz baseline ===")
    for target in args.targets:
        rows = by_target.get(target, [])
        if not rows:
            print(f"{target}: no successful runs")
            continue
        b2c = rows[0].get("baseline_B2C_roc_auc")
        b2a = rows[0].get("baseline_B2A_roc_auc")
        print(f"\n{target}  (raw Boltz: B2C AUC={b2c}, B2A AUC={b2a})")
        for metric, higher in HEADLINE:
            top = _best(rows, metric, higher)
            if top:
                print(f"  best {metric:<28} = {top[metric]:<8} via {top['combo']}")

    print(f"\n[sweep] wrote summary.json, summary_by_target.json, medians.json to {args.out_root}")


if __name__ == "__main__":
    main()
