"""Part 2: does Boltz-2 (via its affinity embeddings) track peptide mutational
effects on binding?

Joins the extracted embeddings under ``targets/peptides/<system>__<receptor>/``
to the measured affinity tables in
``data/Boltz-2/peptides/<system>/<receptor>/manifest.tsv`` (regenerate those
with ``make_boltz_inputs_{bh3,p53}.py`` if absent) and runs the two evaluations
Part 2 calls for: within-series rank correlation and ΔΔG-sign agreement.

Important framing: we only have the *embeddings*, not Boltz-2's own scalar
affinity output (no peptide affinity JSONs were extracted). So this measures
whether the embeddings carry **learnable / geometric** mutational signal — it is
the embedding-model arm, not the raw-Boltz baseline. Two complementary tests:

  BH3 (n=689 per receptor): supervised embedding regressor, K-fold CV,
      out-of-fold Spearman of predicted vs measured apparent affinity
      (higher = tighter). Reported overall and per background (Bim / PUMA).

  p53 (~10-11 point mutants per scaffold per receptor): too few rows for a
      384-dim supervised model, so the headline is the *model-free* magnitude
      probe — Spearman of embedding shift-from-WT vs measured |ΔΔG|. A
      leave-one-out Ridge Spearman + ΔΔG-sign agreement are reported as
      n-limited secondary numbers. Truncation analogs are scored separately
      from point substitutions.

Embedding view: ``head_mean`` — the ensemble-averaged representation just
before the scalar affinity heads (the rep whose movement bounds how much the
scalar prediction can move).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
PEPTIDE_EMB_ROOT = REPO_ROOT / "targets" / "peptides"
LABEL_ROOT = REPO_ROOT / "data" / "Boltz-2" / "peptides"
OUT_DIR = REPO_ROOT / "runs" / "peptide_embeddings"

EMB_KEY = "head_mean"
RIDGE_ALPHAS = np.logspace(-2, 5, 24)


def _load_embeddings(target_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path in sorted(target_dir.glob("affinity_*.npz")):
        arr = np.load(path, allow_pickle=True)
        out[str(arr["peptide_id"])] = np.asarray(arr[EMB_KEY], dtype=np.float64).reshape(-1)
    return out


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _ridge_oof(X: np.ndarray, y: np.ndarray, splitter) -> np.ndarray:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    return cross_val_predict(model, X, y, cv=splitter)


# --------------------------------------------------------------------------- #
# BH3: supervised CV Spearman per receptor
# --------------------------------------------------------------------------- #
def analyze_bh3(receptor: str) -> dict:
    emb = _load_embeddings(PEPTIDE_EMB_ROOT / f"bh3__{receptor}")
    manifest = _read_tsv(LABEL_ROOT / "bh3" / receptor / "manifest.tsv")

    ids, X, y, bg = [], [], [], []
    for row in manifest:
        pid = row["peptide_id"]
        val = row["apparent_value"]
        if pid not in emb or not val:
            continue
        ids.append(pid)
        X.append(emb[pid])
        y.append(float(val))
        bg.append(row["bg"])
    X = np.vstack(X)
    y = np.asarray(y)
    bg = np.asarray(bg)

    oof = _ridge_oof(X, y, KFold(n_splits=5, shuffle=True, random_state=0))
    rho, p = spearmanr(oof, y)
    r, _ = pearsonr(oof, y)

    # Within-background reads. (a) pooled = Spearman of the pooled model's OOF
    # preds restricted to this background. (b) within_cv = a model cross-
    # validated *only* within this background, so the within-series ranking
    # borrows no cross-background signal (the honest within-series number).
    per_bg = {}
    for b in sorted(set(bg)):
        m = bg == b
        if m.sum() < 10:
            continue
        pooled_rho, _ = spearmanr(oof[m], y[m])
        within = _ridge_oof(X[m], y[m],
                            KFold(n_splits=5, shuffle=True, random_state=0))
        within_rho, within_p = spearmanr(within, y[m])
        per_bg[b] = {
            "n": int(m.sum()),
            "spearman_pooled_oof": float(pooled_rho),
            "spearman_within_cv": float(within_rho),
            "spearman_within_cv_p": float(within_p),
        }

    return {
        "receptor": receptor,
        "n": int(len(y)),
        "label": "apparent_value (higher=tighter)",
        "cv": "KFold(5, shuffle); per-bg = within-background KFold(5)",
        "spearman_oof_vs_measured": float(rho),
        "spearman_p": float(p),
        "pearson_oof_vs_measured": float(r),
        "per_background": per_bg,
    }


# --------------------------------------------------------------------------- #
# p53: model-free magnitude probe + n-limited supervised
# --------------------------------------------------------------------------- #
def analyze_p53(receptor: str) -> dict:
    emb = _load_embeddings(PEPTIDE_EMB_ROOT / f"p53__{receptor}")
    manifest = _read_tsv(LABEL_ROOT / "p53" / receptor / "manifest.tsv")

    scaffolds: dict[str, dict] = {}
    for row in manifest:
        sc = row["scaffold"]
        scaffolds.setdefault(sc, {"wt": None, "rows": []})
        if row["mutation_label"] == "WT":
            scaffolds[sc]["wt"] = row
        scaffolds[sc]["rows"].append(row)

    results = {}
    for sc, blk in scaffolds.items():
        wt = blk["wt"]
        if wt is None or wt["peptide_id"] not in emb:
            continue
        wt_vec = emb[wt["peptide_id"]]
        wt_pkd = -math.log10(float(wt["kd_M"]))

        def collect(classes: set[str]):
            d_list, ddg_list, pkd_list, vecs = [], [], [], []
            for row in blk["rows"]:
                if row["analog_class"] not in classes:
                    continue
                if not row["ddG_kcal_per_mol"] or not row["kd_M"]:
                    continue  # skip not_determined
                pid = row["peptide_id"]
                if pid not in emb:
                    continue
                v = emb[pid]
                d_list.append(float(np.linalg.norm(v - wt_vec)))
                ddg_list.append(float(row["ddG_kcal_per_mol"]))
                pkd_list.append(-math.log10(float(row["kd_M"])))
                vecs.append(v)
            return (np.array(d_list), np.array(ddg_list),
                    np.array(pkd_list), np.array(vecs))

        block_out = {}

        # --- point substitutions (Ala-scan) ---
        for label, classes in [("point_mutants", {"ala_scan"}),
                               ("point_plus_trunc", {"ala_scan", "truncation"})]:
            d, ddg, pkd, vecs = collect(classes)
            if len(d) < 4:
                continue
            # model-free magnitude probe: embedding shift vs |ΔΔG|
            mag_rho, mag_p = spearmanr(d, np.abs(ddg))
            entry = {
                "n": int(len(d)),
                "magnitude_probe_spearman_dist_vs_absDDG": float(mag_rho),
                "magnitude_probe_p": float(mag_p),
            }
            # n-limited supervised: LOO Ridge over {mutants + WT} -> pKd. WT is
            # appended so it is held out in its own fold and serves as a genuine
            # reference for the ΔΔG-sign test below.
            if len(d) >= 6:
                X_all = np.vstack([vecs, wt_vec[None, :]])
                pkd_all = np.append(pkd, wt_pkd)
                oof_all = _ridge_oof(X_all, pkd_all, LeaveOneOut())
                mut_oof, wt_oof = oof_all[:-1], oof_all[-1]
                srho, sp = spearmanr(mut_oof, pkd)
                # WT-anchored ΔΔG-sign agreement: predicted destabilization
                # (pKd_WT_heldout - pKd_mut) vs measured ddG (positive=weaker).
                pred_ddg = wt_oof - mut_oof
                agree = np.sign(pred_ddg) == np.sign(ddg)
                strong = np.abs(ddg) >= 1.0  # clear effects, above assay noise
                entry.update({
                    "supervised_loo_spearman": float(srho),
                    "supervised_loo_p": float(sp),
                    "ddg_sign_agreement": float(np.mean(agree)),
                    "ddg_sign_n": int(len(agree)),
                    "ddg_sign_agreement_strong": (
                        float(np.mean(agree[strong])) if strong.any() else None),
                    "ddg_sign_n_strong": int(strong.sum()),
                })
            block_out[label] = entry

        results[sc] = block_out

    return {"receptor": receptor, "scaffolds": results,
            "label": "pKd = -log10(kd_M); ddG kcal/mol (positive=weaker)"}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"embedding_key": EMB_KEY, "bh3": {}, "p53": {}}

    print(f"Embedding view: {EMB_KEY}\n")
    print("=== BH3: supervised CV Spearman (embeddings -> apparent affinity) ===")
    print(f"{'receptor':<10}{'n':>5}{'Spearman':>11}{'p':>11}{'Pearson':>10}"
          f"   within-bg CV Spearman")
    for receptor in ("Bcl-xL", "Mcl-1", "Bfl-1"):
        res = analyze_bh3(receptor)
        out["bh3"][receptor] = res
        bg = "  ".join(f"{b}:{v['spearman_within_cv']:.3f}(n{v['n']})"
                       for b, v in res["per_background"].items())
        print(f"{receptor:<10}{res['n']:>5}{res['spearman_oof_vs_measured']:>11.3f}"
              f"{res['spearman_p']:>11.2e}{res['pearson_oof_vs_measured']:>10.3f}   {bg}")

    print("\n=== p53: model-free magnitude probe + n-limited supervised ===")
    print("(magnitude probe = Spearman of embedding shift-from-WT vs measured |ddG|)")
    for receptor in ("MDM2", "MDMX"):
        res = analyze_p53(receptor)
        out["p53"][receptor] = res
        for sc, blocks in res["scaffolds"].items():
            for label, e in blocks.items():
                if "supervised_loo_spearman" in e:
                    strong = ("n/a" if e["ddg_sign_agreement_strong"] is None
                              else f"{e['ddg_sign_agreement_strong']:.2f}")
                    sup = (f"  | LOO rho={e['supervised_loo_spearman']:.3f}"
                           f" ddG-sign={e['ddg_sign_agreement']:.2f}"
                           f" (|ddG|>=1: {strong} on n={e['ddg_sign_n_strong']})")
                else:
                    sup = ""
                print(f"  {receptor}/{sc}/{label} (n={e['n']}): "
                      f"magnitude rho={e['magnitude_probe_spearman_dist_vs_absDDG']:.3f} "
                      f"(p={e['magnitude_probe_p']:.3f}){sup}")

    out_path = OUT_DIR / "part2_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
