"""Part 2 follow-ups that need no new data (embeddings + measurements only):

(A) Embedding-key sweep — which view (pair_mean / head_ens1 / head_ens2 /
    head_mean / pair_mean+head_mean) best tracks mutational effects, per target.
    BH3 metric = pooled KFold-5 CV Spearman; p53 metric = model-free magnitude
    probe (embedding shift-from-WT vs measured |ΔΔG|, point mutants pooled).

(B) Replicate noise ceiling (BH3) — test-retest Spearman between the main and
    replicate SORTCERY sorts on the cross-target peptides. This is the
    reproducibility ceiling: no model can rank better than the assay agrees with
    itself. The embedding-model Spearman (head_mean) is printed alongside.

(C) Cross-target selectivity (BH3) — the 689 peptides are folded against all
    three receptors, so we can ask whether the embeddings capture *relative*
    receptor preference. Affinities are rank-normalised within each receptor
    first (SORTCERY apparent values are only internally consistent per target),
    then selectivity = percentile[r1] - percentile[r2]; we report Spearman of
    embedding-predicted vs measured selectivity for each receptor pair.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import KFold

# reuse helpers/paths from the main Part-2 script
from part2_analysis import (LABEL_ROOT, OUT_DIR, PEPTIDE_EMB_ROOT, _read_tsv,
                            _ridge_oof)

BH3_RECEPTORS = ("Bcl-xL", "Mcl-1", "Bfl-1")
P53_RECEPTORS = ("MDM2", "MDMX")
VIEWS = {
    "pair_mean": ["pair_mean"],
    "head_ens1": ["head_ens1"],
    "head_ens2": ["head_ens2"],
    "head_mean": ["head_mean"],
    "pair_mean+head_mean": ["pair_mean", "head_mean"],
}
KFOLD = KFold(n_splits=5, shuffle=True, random_state=0)


def load_arrays(target_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """pid -> {array_key: vector}."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(target_dir.glob("affinity_*.npz")):
        arr = np.load(path, allow_pickle=True)
        out[str(arr["peptide_id"])] = {
            k: np.asarray(arr[k], dtype=np.float64).reshape(-1)
            for k in ("pair_mean", "head_ens1", "head_ens2", "head_mean")
        }
    return out


def build_X(pids, arrays, view_keys):
    return np.vstack([np.concatenate([arrays[p][k] for k in view_keys])
                      for p in pids])


# --------------------------------------------------------------------------- #
# (A) embedding-key sweep
# --------------------------------------------------------------------------- #
def sweep_bh3(receptor: str) -> dict[str, float]:
    arrays = load_arrays(PEPTIDE_EMB_ROOT / f"bh3__{receptor}")
    manifest = _read_tsv(LABEL_ROOT / "bh3" / receptor / "manifest.tsv")
    pids, y = [], []
    for row in manifest:
        if row["peptide_id"] in arrays and row["apparent_value"]:
            pids.append(row["peptide_id"])
            y.append(float(row["apparent_value"]))
    y = np.asarray(y)
    res = {}
    for view, keys in VIEWS.items():
        X = build_X(pids, arrays, keys)
        oof = _ridge_oof(X, y, KFOLD)
        res[view] = float(spearmanr(oof, y)[0])
    return res


def sweep_p53(receptor: str) -> dict[str, float]:
    arrays = load_arrays(PEPTIDE_EMB_ROOT / f"p53__{receptor}")
    manifest = _read_tsv(LABEL_ROOT / "p53" / receptor / "manifest.tsv")
    by_scaffold: dict[str, dict] = defaultdict(lambda: {"wt": None, "mut": []})
    for row in manifest:
        sc = row["scaffold"]
        if row["mutation_label"] == "WT":
            by_scaffold[sc]["wt"] = row["peptide_id"]
        elif row["analog_class"] == "ala_scan" and row["ddG_kcal_per_mol"]:
            by_scaffold[sc]["mut"].append(row)
    res = {}
    for view, keys in VIEWS.items():
        dists, absddg = [], []
        for sc, blk in by_scaffold.items():
            if blk["wt"] is None or blk["wt"] not in arrays:
                continue
            wt = np.concatenate([arrays[blk["wt"]][k] for k in keys])
            for row in blk["mut"]:
                pid = row["peptide_id"]
                if pid not in arrays:
                    continue
                v = np.concatenate([arrays[pid][k] for k in keys])
                dists.append(float(np.linalg.norm(v - wt)))
                absddg.append(abs(float(row["ddG_kcal_per_mol"])))
        res[view] = float(spearmanr(dists, absddg)[0])
    return res


# --------------------------------------------------------------------------- #
# (B) replicate noise ceiling
# --------------------------------------------------------------------------- #
def replicate_ceiling() -> dict[str, dict]:
    # cross-target peptide set (the 689 we have embeddings for)
    index = _read_tsv(LABEL_ROOT / "bh3" / "peptide_index.tsv")
    cross_seqs = {r["peptide_seq"] for r in index}

    # mean apparent_value per (target, concentration, is_replicate, seq).
    # Concentration matters: Bcl-xL has both a 1 nM (x1) and a 100 nM (x100)
    # main sort, but the replicate is 1 nM only — pooling concentrations
    # deflates the ceiling, so test-retest must be computed within concentration.
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with (LABEL_ROOT.parents[1] / "source" / "bh3" / "measurements.tsv").open(
            encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["is_pilot"] == "True" or row["peptide_seq"] not in cross_seqs:
                continue
            key = (row["target"], row["concentration_nM"],
                   row["is_replicate"] == "True")
            buckets[key][row["peptide_seq"]].append(float(row["apparent_value"]))

    out = {}
    for receptor in BH3_RECEPTORS:
        concs = sorted({k[1] for k in buckets if k[0] == receptor})
        per_conc = {}
        for conc in concs:
            m = buckets.get((receptor, conc, False), {})
            r = buckets.get((receptor, conc, True), {})
            shared = sorted(set(m) & set(r))
            if len(shared) < 10:
                continue
            mv = [np.mean(m[s]) for s in shared]
            rv = [np.mean(r[s]) for s in shared]
            per_conc[conc] = {"n": len(shared),
                              "spearman": float(spearmanr(mv, rv)[0])}
        # headline = concentration-matched pair with the most shared peptides
        best = max(per_conc.items(), key=lambda kv: kv[1]["n"], default=None)
        out[receptor] = {
            "ceiling_spearman": best[1]["spearman"] if best else None,
            "ceiling_concentration_nM": best[0] if best else None,
            "n": best[1]["n"] if best else 0,
            "per_concentration": per_conc,
        }
    return out


# --------------------------------------------------------------------------- #
# (C) cross-target selectivity
# --------------------------------------------------------------------------- #
def selectivity() -> dict:
    # per-receptor: out-of-fold predicted + measured, aligned by peptide_id
    measured, predicted, pid_order = {}, {}, None
    for receptor in BH3_RECEPTORS:
        arrays = load_arrays(PEPTIDE_EMB_ROOT / f"bh3__{receptor}")
        manifest = _read_tsv(LABEL_ROOT / "bh3" / receptor / "manifest.tsv")
        pids, y = [], []
        for row in manifest:
            if row["peptide_id"] in arrays and row["apparent_value"]:
                pids.append(row["peptide_id"])
                y.append(float(row["apparent_value"]))
        y = np.asarray(y)
        X = build_X(pids, arrays, VIEWS["head_mean"])
        oof = _ridge_oof(X, y, KFOLD)
        # rank-normalise within receptor (apparent values only rank-consistent
        # per target), 0..1 percentile
        measured[receptor] = dict(zip(pids, rankdata(y) / len(y)))
        predicted[receptor] = dict(zip(pids, rankdata(oof) / len(oof)))
        pid_order = pids if pid_order is None else pid_order

    out = {}
    for r1, r2 in [("Mcl-1", "Bcl-xL"), ("Bfl-1", "Bcl-xL"), ("Mcl-1", "Bfl-1")]:
        shared = [p for p in measured[r1] if p in measured[r2]]
        meas_sel = np.array([measured[r1][p] - measured[r2][p] for p in shared])
        pred_sel = np.array([predicted[r1][p] - predicted[r2][p] for p in shared])
        rho, p = spearmanr(pred_sel, meas_sel)
        out[f"{r1}_vs_{r2}"] = {"n": len(shared), "spearman": float(rho),
                               "p": float(p)}
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}

    print("=== (A) embedding-key sweep ===")
    print(f"{'target':<16}" + "".join(f"{v:>20}" for v in VIEWS))
    out["embedding_key_sweep"] = {}
    for receptor in BH3_RECEPTORS:
        res = sweep_bh3(receptor)
        out["embedding_key_sweep"][f"bh3__{receptor}"] = res
        print(f"{'bh3__'+receptor:<16}" + "".join(f"{res[v]:>20.3f}" for v in VIEWS))
    for receptor in P53_RECEPTORS:
        res = sweep_p53(receptor)
        out["embedding_key_sweep"][f"p53__{receptor}"] = res
        print(f"{'p53__'+receptor:<16}" + "".join(f"{res[v]:>20.3f}" for v in VIEWS))
    print("(BH3 = CV Spearman; p53 = magnitude-probe Spearman)")

    print("\n=== (B) BH3 replicate noise ceiling vs model ===")
    ceiling = replicate_ceiling()
    out["replicate_ceiling"] = ceiling
    model = {"Bcl-xL": 0.657, "Mcl-1": 0.766, "Bfl-1": 0.791}  # head_mean pooled CV
    for receptor in BH3_RECEPTORS:
        e = ceiling[receptor]
        c = e["ceiling_spearman"]
        cs = f"{c:.3f}" if c is not None else "n/a"
        print(f"  {receptor:<8} ceiling={cs} (@{e['ceiling_concentration_nM']}nM, "
              f"n={e['n']})   model(head_mean CV)={model[receptor]:.3f}")

    print("\n=== (C) cross-target selectivity (predicted vs measured, rank-normalised) ===")
    sel = selectivity()
    out["selectivity"] = sel
    for pair, e in sel.items():
        print(f"  {pair:<18} Spearman={e['spearman']:.3f} (p={e['p']:.2e}, n={e['n']})")

    out_path = OUT_DIR / "part2_extras.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
