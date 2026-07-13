"""Label-free Part-2 diagnostic: does the Boltz-2 affinity embedding move
under peptide mutation?

This is the *embedding-geometry* half of Part 2 — it needs no measured
affinities, only the extracted ``affinity_*.npz`` files under
``data/peptides/modeling/features/boltz_embeddings/<system>__<receptor>/``. It
answers the question the Rognan
paper raised (Boltz-2 affinity is largely insensitive to binding-site
mutations) on the *ligand* side: if the representation feeding the scalar
heads barely changes across a mutational series, the scalar prediction cannot
track the mutation either.

It does NOT compute within-series Spearman or ΔΔG-sign agreement — those need
``data/peptides/source/<system>/measurements.tsv`` (kd_M / ddG columns) and, for BH3,
``peptide_index.tsv`` to map opaque ``bh3_NNNN`` ids to sequences. This script
deliberately remains label-free and reports only what the embeddings support,
plus a QC summary (counts, dims, degeneracy check).

Each npz holds: pair_mean (128), head_ens1 (384), head_ens2 (384),
head_mean (384), peptide_id, target. ``head_mean`` is the ensemble-averaged
post-MLP representation immediately before the scalar affinity heads, so it is
the most relevant view for "can the prediction move?".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPTIDE_ROOT = (
    REPO_ROOT / "data" / "peptides" / "modeling" / "features" / "boltz_embeddings"
)
OUT_DIR = REPO_ROOT / "runs" / "peptide_embeddings"

# Representation used for the sensitivity probe. head_mean = ensemble-averaged
# rep just before the scalar affinity heads.
PRIMARY_KEY = "head_mean"


def _load_target(target_dir: Path) -> tuple[list[str], np.ndarray]:
    """Return (peptide_ids, matrix[n, d]) of the PRIMARY_KEY embedding."""
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for path in sorted(target_dir.glob("affinity_*.npz")):
        arr = np.load(path, allow_pickle=True)
        ids.append(str(arr["peptide_id"]))
        vecs.append(np.asarray(arr[PRIMARY_KEY], dtype=np.float64).reshape(-1))
    return ids, np.vstack(vecs)


def _p53_scaffold(peptide_id: str) -> str | None:
    if peptide_id.startswith("PMI_"):
        return "PMI"
    if peptide_id.startswith("p53_17_28_"):
        return "p53_17_28"
    return None


def _wt_distances(ids: list[str], mat: np.ndarray) -> dict:
    """For p53 targets: per-mutant L2 distance and cosine from the scaffold WT.

    Returns {} for targets whose ids carry no recoverable WT (e.g. BH3).
    """
    scaffolds: dict[str, list[int]] = {}
    for i, pid in enumerate(ids):
        sc = _p53_scaffold(pid)
        if sc is not None:
            scaffolds.setdefault(sc, []).append(i)
    if not scaffolds:
        return {}

    out: dict = {}
    for sc, idx in scaffolds.items():
        wt_pos = next((i for i in idx if ids[i] == f"{sc}_WT"), None)
        if wt_pos is None:
            continue
        wt = mat[wt_pos]
        wt_norm = float(np.linalg.norm(wt))
        rows = []
        for i in idx:
            if i == wt_pos:
                continue
            v = mat[i]
            l2 = float(np.linalg.norm(v - wt))
            denom = np.linalg.norm(v) * wt_norm
            cos = float(v @ wt / denom) if denom > 0 else float("nan")
            rows.append({
                "peptide_id": ids[i],
                "l2_from_wt": l2,
                "l2_rel_wt_norm": l2 / wt_norm if wt_norm > 0 else float("nan"),
                "cosine_to_wt": cos,
            })
        rows.sort(key=lambda r: r["l2_from_wt"], reverse=True)
        l2s = np.array([r["l2_from_wt"] for r in rows])
        out[sc] = {
            "wt_id": f"{sc}_WT",
            "wt_norm": wt_norm,
            "n_mutants": len(rows),
            "l2_from_wt_min": float(l2s.min()),
            "l2_from_wt_max": float(l2s.max()),
            "l2_from_wt_mean": float(l2s.mean()),
            "l2_rel_wt_norm_mean": float(np.mean([r["l2_rel_wt_norm"] for r in rows])),
            "per_mutant": rows,
        }
    return out


def _spread(mat: np.ndarray) -> dict:
    """Global spread of the embedding set — QC + degeneracy check."""
    centroid = mat.mean(axis=0)
    dists = np.linalg.norm(mat - centroid, axis=1)
    # mean pairwise distance via a small sample to stay cheap on large sets
    n = mat.shape[0]
    rng = np.random.default_rng(0)
    k = min(n, 200)
    sample = mat[rng.choice(n, size=k, replace=False)] if n > k else mat
    pair = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    iu = np.triu_indices(sample.shape[0], k=1)
    pair_vals = pair[iu]
    return {
        "centroid_norm": float(np.linalg.norm(centroid)),
        "dist_to_centroid_mean": float(dists.mean()),
        "dist_to_centroid_std": float(dists.std()),
        "mean_pairwise_l2": float(pair_vals.mean()),
        "min_pairwise_l2": float(pair_vals.min()),
        "total_variance": float(((mat - centroid) ** 2).sum(axis=1).mean()),
        "n_unique_rows": int(np.unique(mat, axis=0).shape[0]),
    }


def main() -> int:
    if not PEPTIDE_ROOT.exists():
        raise SystemExit(f"missing {PEPTIDE_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = sorted(d for d in PEPTIDE_ROOT.iterdir() if d.is_dir())
    summary: dict[str, dict] = {}

    print(f"Primary embedding key: {PRIMARY_KEY}\n")
    header = f"{'target':<16}{'n':>6}{'dim':>6}{'uniq':>7}{'meanPairL2':>12}{'minPairL2':>11}"
    print(header)
    print("-" * len(header))

    for tdir in targets:
        ids, mat = _load_target(tdir)
        spread = _spread(mat)
        wt = _wt_distances(ids, mat)
        summary[tdir.name] = {
            "n_peptides": len(ids),
            "embedding_dim": int(mat.shape[1]),
            "spread": spread,
            "wt_distances": wt,
        }
        print(f"{tdir.name:<16}{len(ids):>6}{mat.shape[1]:>6}"
              f"{spread['n_unique_rows']:>7}{spread['mean_pairwise_l2']:>12.4f}"
              f"{spread['min_pairwise_l2']:>11.4f}")

    # p53 WT-distance detail
    any_wt = False
    for tname, s in summary.items():
        if not s["wt_distances"]:
            continue
        any_wt = True
        print(f"\n=== {tname}: per-mutant shift from WT (head_mean) ===")
        for sc, d in s["wt_distances"].items():
            print(f"  scaffold {sc} (WT |v|={d['wt_norm']:.3f}, n={d['n_mutants']}): "
                  f"L2 from WT min={d['l2_from_wt_min']:.4f} "
                  f"mean={d['l2_from_wt_mean']:.4f} max={d['l2_from_wt_max']:.4f} "
                  f"(mean rel to |WT| = {d['l2_rel_wt_norm_mean']:.3%})")
            top = d["per_mutant"][:5]
            for r in top:
                print(f"      {r['peptide_id']:<18} L2={r['l2_from_wt']:.4f} "
                      f"cos={r['cosine_to_wt']:.5f}")
    if not any_wt:
        print("\n(no WT-resolvable series — BH3 ids are opaque without peptide_index.tsv)")

    out_path = OUT_DIR / "embedding_sensitivity.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
