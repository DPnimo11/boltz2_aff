"""Part 2, open item #1: the raw Boltz-2 scalar baseline (B2-A / B2-C).

This is the apples-to-apples Rognan comparison — does Boltz-2's *own* scalar
affinity output track peptide mutational effects, with no model fitting? It is
the counterpart to `part2_analysis.py` (the embedding-model arm); reporting both
side by side answers whether the mutational signal that is clearly present in
the embeddings survives into the published scalar.

Requires the peptide cofolding to have been run. The 2139 input YAMLs already
exist (`scripts/make_boltz_inputs_{bh3,p53}.py`); running Boltz-2 over them is an
external GPU job (same pipeline / `../boltz` fork that produced Part 1's JSONs).
Expected output:

    data/peptides/boltz/outputs/<system>/<receptor>/<peptide_id>/affinity_<peptide_id>.json

with fields `affinity_pred_value` (B2-A, lower = stronger binder) and
`affinity_probability_binary` (B2-C, 0-1, higher = stronger). When no JSONs are
found the script reports what is missing and exits cleanly.

Scores are oriented so higher = tighter to match the measured labels:
    score_A = -affinity_pred_value     score_C = affinity_probability_binary
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = REPO_ROOT / "data" / "peptides" / "boltz" / "inputs"
BOLTZ_OUTPUT_ROOT = REPO_ROOT / "data" / "peptides" / "boltz" / "outputs"
EMB_RESULTS = REPO_ROOT / "runs" / "peptide_embeddings" / "part2_results.json"
OUT_DIR = REPO_ROOT / "runs" / "peptide_embeddings"
BH3_RECEPTORS = ("Bcl-xL", "Mcl-1", "Bfl-1")
P53_RECEPTORS = ("MDM2", "MDMX")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_raw_scalars(system: str, receptor: str) -> dict[str, dict[str, float]]:
    """peptide_id -> {score_A, score_C} from the cofolded affinity JSONs."""
    base = BOLTZ_OUTPUT_ROOT / system / receptor
    out: dict[str, dict[str, float]] = {}
    for path in sorted(base.rglob("affinity_*.json")):
        if path.name.startswith("confidence_") or path.parent.name == "input":
            continue
        pid = path.stem[len("affinity_"):]
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        if "affinity_pred_value" not in payload:
            continue
        out[pid] = {
            "score_A": -float(payload["affinity_pred_value"]),
            "score_C": float(payload.get("affinity_probability_binary", float("nan"))),
        }
    return out


def _emb_reference() -> dict:
    if EMB_RESULTS.exists():
        return json.loads(EMB_RESULTS.read_text(encoding="utf-8"))
    return {}


def analyze_bh3(receptor: str) -> dict | None:
    scalars = load_raw_scalars("bh3", receptor)
    if not scalars:
        return None
    manifest = _read_tsv(LABEL_ROOT / "bh3" / receptor / "manifest.tsv")
    a, c, y = [], [], []
    for row in manifest:
        pid = row["peptide_id"]
        if pid in scalars and row["apparent_value"]:
            a.append(scalars[pid]["score_A"])
            c.append(scalars[pid]["score_C"])
            y.append(float(row["apparent_value"]))
    y = np.asarray(y)
    return {
        "n": int(len(y)),
        "raw_B2A_spearman": float(spearmanr(a, y)[0]),
        "raw_B2C_spearman": float(spearmanr(c, y)[0]),
    }


def analyze_p53(receptor: str) -> dict | None:
    scalars = load_raw_scalars("p53", receptor)
    if not scalars:
        return None
    manifest = _read_tsv(LABEL_ROOT / "p53" / receptor / "manifest.tsv")
    by_sc: dict[str, dict] = {}
    for row in manifest:
        by_sc.setdefault(row["scaffold"], {"wt": None, "mut": []})
        if row["mutation_label"] == "WT":
            by_sc[row["scaffold"]]["wt"] = row
        elif row["analog_class"] == "ala_scan" and row["ddG_kcal_per_mol"]:
            by_sc[row["scaffold"]]["mut"].append(row)

    out = {}
    for sc, blk in by_sc.items():
        wt = blk["wt"]
        if wt is None or wt["peptide_id"] not in scalars:
            continue
        wt_s = scalars[wt["peptide_id"]]
        a, c, pkd, ddg = [], [], [], []
        for row in blk["mut"]:
            pid = row["peptide_id"]
            if pid not in scalars:
                continue
            a.append(scalars[pid]["score_A"])
            c.append(scalars[pid]["score_C"])
            pkd.append(-math.log10(float(row["kd_M"])))
            ddg.append(float(row["ddG_kcal_per_mol"]))
        if len(a) < 4:
            continue
        a, c, pkd, ddg = map(np.asarray, (a, c, pkd, ddg))
        # WT-anchored ΔΔG-sign: predicted destabilization (score_WT - score_mut)
        # vs measured ddG (positive = weaker); strong subset |ddG| >= 1 kcal/mol.
        strong = np.abs(ddg) >= 1.0

        def sign_block(scores, wt_score):
            pred_ddg = wt_score - scores
            agree = np.sign(pred_ddg) == np.sign(ddg)
            return {
                "spearman_vs_pKd": float(spearmanr(scores, pkd)[0]),
                "ddg_sign_agreement": float(np.mean(agree)),
                "ddg_sign_agreement_strong": (
                    float(np.mean(agree[strong])) if strong.any() else None),
                "ddg_sign_n_strong": int(strong.sum()),
            }

        out[sc] = {
            "n": int(len(a)),
            "B2A": sign_block(a, wt_s["score_A"]),
            "B2C": sign_block(c, wt_s["score_C"]),
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emb = _emb_reference()
    out = {"bh3": {}, "p53": {}}

    found = any(load_raw_scalars(s, r)
                for s, recs in (("bh3", BH3_RECEPTORS), ("p53", P53_RECEPTORS))
                for r in recs)
    if not found:
        print("No peptide affinity JSONs found yet.\n")
        print("The raw-Boltz scalar baseline needs the cofolding to have been run.")
        print("Inputs are ready under data/peptides/boltz/inputs/*/*/input/.")
        print("Run Boltz-2 over them (external GPU job, ../boltz fork) so outputs land at:")
        print("  data/peptides/boltz/outputs/<system>/<receptor>/<pid>/affinity_<pid>.json")
        print("then re-run this script - it will compute B2-A / B2-C within-series")
        print("Spearman + ddG-sign and compare against the embedding-model arm.")
        return 0

    print("=== BH3: raw Boltz-2 scalar within-series Spearman vs apparent affinity ===")
    print(f"{'receptor':<10}{'n':>5}{'B2-A':>9}{'B2-C':>9}{'  embed(CV)':>13}")
    for receptor in BH3_RECEPTORS:
        res = analyze_bh3(receptor)
        if res is None:
            continue
        out["bh3"][receptor] = res
        emb_rho = emb.get("bh3", {}).get(receptor, {}).get("spearman_oof_vs_measured")
        es = f"{emb_rho:.3f}" if emb_rho is not None else "n/a"
        print(f"{receptor:<10}{res['n']:>5}{res['raw_B2A_spearman']:>9.3f}"
              f"{res['raw_B2C_spearman']:>9.3f}{es:>13}")

    print("\n=== p53: raw Boltz-2 scalar (Spearman vs pKd; ddG-sign on |ddG|>=1) ===")
    for receptor in P53_RECEPTORS:
        res = analyze_p53(receptor)
        if not res:
            continue
        out["p53"][receptor] = res
        for sc, e in res.items():
            for arm in ("B2A", "B2C"):
                b = e[arm]
                strong = ("n/a" if b["ddg_sign_agreement_strong"] is None
                          else f"{b['ddg_sign_agreement_strong']:.2f}")
                print(f"  {receptor}/{sc}/{arm} (n={e['n']}): "
                      f"rho_vs_pKd={b['spearman_vs_pKd']:.3f}  "
                      f"ddG-sign(|ddG|>=1)={strong} on n={b['ddg_sign_n_strong']}")

    out_path = OUT_DIR / "part2_raw_boltz.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
