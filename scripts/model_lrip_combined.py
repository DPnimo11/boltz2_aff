"""Does LRIP add anything on top of Boltz? Paired-increment per-target test.

For each target, evaluate feature stacks with the same RF / StratifiedGroupKFold
methodology as everything else (``boltz2_aff.modeling.train_classifier``) and
report the *increment* from adding the LRIP block. Each increment pair is scored
on identical rows so the delta is attributable only to the LRIP features:

- ``boltz``          -> ``boltz + lrip``           (LRIP on top of raw Boltz scalars)
- ``combined``       -> ``combined + lrip``        (LRIP on top of embeddings + Boltz)
  where ``combined`` = Boltz affinity embeddings + Boltz scalars.

Outputs to ``runs/lrip_combined/``: per-(target, stack) ``train_classifier``
artifacts, plus ``summary.tsv`` / ``summary_by_target.json`` / ``summary.json``
with per-target AUCs, deltas, and paper-style medians.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from boltz2_aff.features import discover_boltz_scalar_frame, discover_embedding_frame
from boltz2_aff.modeling import train_classifier

# Reuse the parser + label/id handling from the standalone LRIP script.
from model_lrip import (  # type: ignore
    ALL_TARGETS,
    BOLTZ_SCALARS,
    EMBEDDING_ROOT,
    LIGAND_ID_FIXES,
    LRIP_DIR,
    load_labels,
    parse_lrip,
)

OUT_ROOT = Path(__file__).resolve().parents[1] / "runs/lrip_combined"


def base_frame(target: str, labels: pd.DataFrame, boltz: pd.DataFrame,
               embeddings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    lrip, lrip_cols = parse_lrip(LRIP_DIR / f"{target}.dat")
    lrip["ligand_id"] = lrip["ligand_id"].replace(LIGAND_ID_FIXES.get(target, {}))
    lrip["target"] = target

    frame = labels[labels["target"] == target].merge(lrip, on=["target", "ligand_id"], how="inner")
    if len(frame) != len(lrip):
        raise ValueError(f"{target}: LRIP rows failed to join labels")

    boltz_cols = [c for c in boltz.columns if c.startswith("boltz_")]
    frame = frame.merge(boltz[["target", "variant", "ligand_id", *boltz_cols]],
                        on=["target", "variant", "ligand_id"], how="left")

    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    frame = frame.merge(embeddings[["target", "ligand_id", *emb_cols]],
                        on=["target", "ligand_id"], how="left")

    frame["group_id"] = frame["target"].astype(str) + "::" + frame["ligand_id"].astype(str)
    return frame, {"lrip": lrip_cols, "boltz": boltz_cols, "emb": emb_cols}


def auc(frame: pd.DataFrame, cols: list[str], out_dir: Path) -> float | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    return train_classifier(frame, cols, out_dir, max_splits=5, random_state=42).get("cv_roc_auc")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    boltz = discover_boltz_scalar_frame(BOLTZ_SCALARS, variants=["wt"])
    embeddings = discover_embedding_frame([EMBEDDING_ROOT], variants=["wt"])

    rows: list[dict[str, Any]] = []
    for target in ALL_TARGETS:
        frame, cols = base_frame(target, labels, boltz, embeddings)

        # boltz vs boltz+lrip -- rows with Boltz scalars (full wt coverage).
        has_boltz = frame[cols["boltz"]].notna().any(axis=1)
        fb = frame[has_boltz].copy()
        a_boltz = auc(fb, cols["boltz"], OUT_ROOT / target / "boltz")
        a_boltz_lrip = auc(fb, cols["boltz"] + cols["lrip"], OUT_ROOT / target / "boltz_lrip")

        # combined vs combined+lrip -- rows that also have embeddings.
        has_emb = frame[cols["emb"]].notna().any(axis=1) if cols["emb"] else pd.Series(False, index=frame.index)
        fc = frame[has_boltz & has_emb].copy()
        combined = cols["emb"] + cols["boltz"]
        a_comb = auc(fc, combined, OUT_ROOT / target / "combined") if len(fc) else None
        a_comb_lrip = auc(fc, combined + cols["lrip"], OUT_ROOT / target / "combined_lrip") if len(fc) else None

        row = {
            "target": target,
            "n_boltz_rows": int(len(fb)),
            "n_combined_rows": int(len(fc)),
            "boltz": a_boltz,
            "boltz_lrip": a_boltz_lrip,
            "d_boltz": _delta(a_boltz_lrip, a_boltz),
            "combined": a_comb,
            "combined_lrip": a_comb_lrip,
            "d_combined": _delta(a_comb_lrip, a_comb),
        }
        rows.append(row)
        print(
            f"{target:7s}  boltz {_f(a_boltz)} -> +lrip {_f(a_boltz_lrip)} ({_fd(row['d_boltz'])})   "
            f"combined {_f(a_comb)} -> +lrip {_f(a_comb_lrip)} ({_fd(row['d_combined'])})"
        )

    pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.tsv", sep="\t", index=False)
    (OUT_ROOT / "summary_by_target.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def med(key: str) -> float | None:
        vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and r[key] is not None]
        return float(statistics.median(vals)) if vals else None

    def wins(delta_key: str) -> dict[str, int]:
        ds = [r[delta_key] for r in rows if isinstance(r[delta_key], (int, float)) and r[delta_key] is not None]
        return {"helps": sum(d > 0 for d in ds), "hurts": sum(d < 0 for d in ds), "n": len(ds)}

    summary = {
        "median_boltz": med("boltz"),
        "median_boltz_lrip": med("boltz_lrip"),
        "median_combined": med("combined"),
        "median_combined_lrip": med("combined_lrip"),
        "boltz_increment": {"median_delta": med("d_boltz"), **wins("d_boltz")},
        "combined_increment": {"median_delta": med("d_combined"), **wins("d_combined")},
        "question": "does adding the LRIP block improve on Boltz / embeddings+Boltz?",
        "methodology": "RF, StratifiedGroupKFold, cv_roc_auc; each increment pair on identical rows",
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n== medians across targets ==")
    print(f"  boltz          {_f(summary['median_boltz'])}  ->  +lrip {_f(summary['median_boltz_lrip'])}"
          f"   (median delta {_fd(summary['boltz_increment']['median_delta'])})")
    print(f"  combined       {_f(summary['median_combined'])}  ->  +lrip {_f(summary['median_combined_lrip'])}"
          f"   (median delta {_fd(summary['combined_increment']['median_delta'])})")
    bi, ci = summary["boltz_increment"], summary["combined_increment"]
    print(f"\n  LRIP helps boltz on {bi['helps']}/{bi['n']} targets (hurts {bi['hurts']}); "
          f"helps combined on {ci['helps']}/{ci['n']} (hurts {ci['hurts']}).")
    print(f"\nwrote {OUT_ROOT}/summary.tsv, summary_by_target.json, summary.json")


def _delta(a: float | None, b: float | None) -> float | None:
    return a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def _f(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else "  -  "


def _fd(v: Any) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) and v is not None else "  -  "


if __name__ == "__main__":
    main()
