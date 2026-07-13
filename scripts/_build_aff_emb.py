#!/usr/bin/env python
"""Reconstruct Boltz-2 affinity embeddings post-hoc from the saved trunk z.

Generalizes the original 2-chain peptides script (rec=chain A, lig=chain B) to
the multi-chain SKEMPI systems: the binder and receptor are now *chain groups*
read from ``data/peptide_systems/boltz/inputs/manifest.tsv``. For each predicted
complex it masks the
trunk pair representation z to the binder<->partner interface, pools it into
pair_mean (128-d), and applies the two affinity-head MLPs from boltz2_aff.ckpt
to get head_ens1/head_ens2/head_mean (384-d).

Pooling matches Boltz's AffinityHeadsTransformer (see ../new_affinity.py):
    mask = (lig x recT) | (rec x ligT) | (lig x ligT)   minus the diagonal
    pair_mean = sum(z * mask) / sum(mask)
The mask is asymmetric (binder-binder self-pairs are kept, receptor-receptor are
not), so the binder side matters. Default: binder = the SMALLER chain group by
residue count (override with --binder-side or the manifest `affinity_binder`
column).

Head MLP note: to stay bit-compatible with the original peptides embeddings, the
final ReLU of affinity_out_mlp is OMITTED by default (matching the produced
peptides npz). Pass --final-relu for the faithful Boltz g_head.

Examples:
  python _build_aff_emb.py                       # all systems under _output/
  python _build_aff_emb.py --systems 1VFB_AB_C   # one system
  # Test 1 (validate against the peptides reference), 2-chain fallback:
  python _build_aff_emb.py --output-root ../peptides/_output --systems p53__MDM2 \
         --no-manifest --validate-against ../peptides/_affinity_embeddings/affinity_embeddings.npz
"""
import argparse, csv, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def load_mlps(ckpt_path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)

    def mlp(mod):
        p = f"affinity_module{mod}.affinity_heads.affinity_out_mlp."
        return [(sd[p + "0.weight"].float().numpy(), sd[p + "0.bias"].float().numpy()),
                (sd[p + "2.weight"].float().numpy(), sd[p + "2.bias"].float().numpy())]

    return mlp(1), mlp(2)


def apply_mlp(x, M, final_relu=False):
    (W0, b0), (W2, b2) = M
    h = np.maximum(x @ W0.T + b0, 0.0)     # Linear -> ReLU
    out = h @ W2.T + b2                     # Linear
    if final_relu:                          # optional trailing ReLU (Boltz g_head)
        out = np.maximum(out, 0.0)
    return out


def chain_ranges(structure_npz):
    """chain name -> (start_token, n_tokens) from a processed structure npz."""
    s = np.load(structure_npz, allow_pickle=True)
    ch = s["chains"]
    return {str(c["name"]): (int(c["res_idx"]), int(c["res_num"])) for c in ch}


def group_token_mask(n_tokens, ranges, chains):
    m = np.zeros(n_tokens, bool)
    for c in chains:
        if c in ranges:
            a, k = ranges[c]
            m[a:a + k] = True
    return m


def read_manifest(path):
    """system -> (group1_chains, group2_chains, affinity_binder_chains)."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            g1 = [c for c in row["group1_chains"] if c.isalnum()]
            g2 = [c for c in row["group2_chains"] if c.isalnum()]
            binder = [c for c in (row.get("affinity_binder") or "") if c.isalnum()]
            out[row["system"]] = (g1, g2, binder)
    return out


def decide_binder(g1, g2, binder_override, ranges, side):
    """Return (binder_chains, receptor_chains)."""
    if binder_override:
        binder = binder_override
        rec = [c for c in g1 + g2 if c not in binder]
        return binder, rec
    if side == "group1":
        return g1, g2
    if side == "group2":
        return g2, g1
    # auto: smaller group by residue (token) count is the binder; tie -> group2
    s1 = sum(ranges[c][1] for c in g1 if c in ranges)
    s2 = sum(ranges[c][1] for c in g2 if c in ranges)
    return (g1, g2) if s1 < s2 else (g2, g1)


def build_system(sys_name, sysdir, groups, mlps, args):
    """Return (ids, pair_mean, head1, head2) for one system directory."""
    M1, M2 = mlps
    embs = sorted(glob.glob(f"{sysdir}/**/embeddings_*.npz", recursive=True))
    if not embs:
        print(f"  [{sys_name}] no embeddings_*.npz found — skipping", flush=True)
        return [], [], [], []

    binder = rec = None
    ids, PM, H1, H2 = [], [], [], []
    for f in embs:
        pid = os.path.basename(f).replace("embeddings_", "").replace(".npz", "")
        sts = glob.glob(f"{sysdir}/**/structures/{pid}.npz", recursive=True)
        if not sts:
            print(f"  [{sys_name}] WARN no structure for {pid}", flush=True)
            continue
        ranges = chain_ranges(sts[0])
        z = np.load(f)["z"][0]                       # (N, N, 128)
        N = z.shape[0]

        if binder is None:                            # decide once per system
            if groups is None:                        # 2-chain fallback (no manifest)
                names = sorted(ranges)
                if len(names) != 2:
                    print(f"  [{sys_name}] needs a manifest (>{len(names)} chains) — skipping")
                    return [], [], [], []
                g1, g2, override = [names[0]], [names[1]], []
            else:
                g1, g2, override = groups
            binder, rec = decide_binder(g1, g2, override, ranges, args.binder_side)
            print(f"  [{sys_name}] binder={''.join(binder)} receptor={''.join(rec)}", flush=True)

        lig = group_token_mask(N, ranges, binder)
        rcp = group_token_mask(N, ranges, rec)
        L, R = lig[:, None], rcp[:, None]
        mask = (L & R.T) | (R & L.T) | (L & L.T)
        np.fill_diagonal(mask, False)
        m = mask.astype(np.float32)
        pm = (z * m[:, :, None]).sum((0, 1)) / (m.sum() + 1e-7)   # (128,)

        ids.append(f"{sys_name}::{pid}")
        PM.append(pm)
        H1.append(apply_mlp(pm, M1, args.final_relu))
        H2.append(apply_mlp(pm, M2, args.final_relu))
    return ids, PM, H1, H2


def validate_against(ref_path, ids, PM, H1, H2, cos_threshold=0.99):
    # Criterion is COSINE vs the reference, not bit-exact max|Δ|. Boltz's trunk z is
    # ~1% nondeterministic run-to-run (bf16 AMP + nondeterministic CUDA/cuequivariance
    # reductions), so a correct reconstruction still shows max|Δ| ~ O(1); two identical
    # re-runs of the same input on the same GPU differ by that much. Cosine ~ 1 confirms
    # the pooling + head math; a wrong binder side / MLP drops it well below threshold.
    ref = np.load(ref_path, allow_pickle=True)
    ridx = {str(rid).split("::", 1)[-1]: i for i, rid in enumerate(ref["ids"])}
    hm = (np.array(H1) + np.array(H2)) / 2
    cos_pm, cos_hm, dpm, dhm, n = [], [], 0.0, 0.0, 0
    for k, idd in enumerate(ids):
        pid = idd.split("::", 1)[-1]
        if pid not in ridx:
            continue
        j = ridx[pid]
        a = np.asarray(PM[k], float); b = np.asarray(ref["pair_mean"][j], float)
        ha = np.asarray(hm[k], float); hb = np.asarray(ref["head_mean"][j], float)
        cos_pm.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        cos_hm.append(float(ha @ hb / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-9)))
        dpm = max(dpm, float(np.abs(a - b).max()))
        dhm = max(dhm, float(np.abs(ha - hb).max()))
        n += 1
    if n == 0:
        print(f"\n[validate] no matching ids vs {os.path.basename(ref_path)}")
        return
    cpm_min, chm_min = min(cos_pm), min(cos_hm)
    print(f"\n[validate] matched {n} ids vs {os.path.basename(ref_path)}")
    print(f"[validate] pair_mean: min cos={cpm_min:.5f} mean cos={np.mean(cos_pm):.5f} max|Δ|={dpm:.3e}")
    print(f"[validate] head_mean: min cos={chm_min:.5f} mean cos={np.mean(cos_hm):.5f} max|Δ|={dhm:.3e}")
    ok = cpm_min >= cos_threshold
    print(f"[validate] {'PASS' if ok else 'DIFFERS'} "
          f"(criterion: min cos(pair_mean) >= {cos_threshold:.3f})")
    if not ok:
        print("[validate] check binder side (--binder-side) / --final-relu / MSA provenance")
    else:
        print("[validate] (max|Δ| is O(1) by design — Boltz z is ~1% nondeterministic run-to-run)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--output-root",
        default=os.path.join(REPO_ROOT, "data", "peptide_systems", "boltz", "outputs"),
                    help="boltz output root containing <system>/ subdirs")
    ap.add_argument(
        "--manifest",
        default=os.path.join(
            REPO_ROOT, "data", "peptide_systems", "boltz", "inputs", "manifest.tsv"
        ),
    )
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/boltz2_aff.ckpt"))
    ap.add_argument(
        "--out-dir",
        default=os.path.join(
            REPO_ROOT, "data", "peptide_systems", "modeling", "features", "reconstructed"
        ),
    )
    ap.add_argument("--systems", nargs="*", help="subset of system dirs (default: all)")
    ap.add_argument("--binder-side", choices=("auto", "group1", "group2"), default="auto")
    ap.add_argument("--no-manifest", action="store_true",
                    help="ignore the manifest; use the 2-chain fallback (peptide validation)")
    ap.add_argument("--final-relu", action="store_true",
                    help="apply the trailing ReLU of affinity_out_mlp (faithful g_head)")
    ap.add_argument("--validate-against", metavar="NPZ",
                    help="compare produced pair_mean/head_mean to a reference npz")
    ap.add_argument("--cos-threshold", type=float, default=0.99,
                    help="min cosine(pair_mean) vs reference to PASS --validate-against (default 0.99; "
                         "bit-exact match is impossible — Boltz z is ~1%% nondeterministic)")
    args = ap.parse_args()

    manifest = {} if args.no_manifest else read_manifest(args.manifest)
    mlps = load_mlps(args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)

    systems = args.systems or sorted(
        d for d in os.listdir(args.output_root)
        if os.path.isdir(os.path.join(args.output_root, d)) and not d.startswith("_"))

    all_ids, all_pm, all_h1, all_h2 = [], [], [], []
    for sys_name in systems:
        sysdir = os.path.join(args.output_root, sys_name)
        if not os.path.isdir(sysdir):
            print(f"  [{sys_name}] no output dir — skipping"); continue
        groups = None if args.no_manifest else manifest.get(sys_name)
        if groups is None and not args.no_manifest:
            print(f"  [{sys_name}] not in manifest — using 2-chain fallback")
        ids, PM, H1, H2 = build_system(sys_name, sysdir, groups, mlps, args)
        if not ids:
            continue
        PM, H1, H2 = np.array(PM, np.float32), np.array(H1, np.float32), np.array(H2, np.float32)
        # per-system file
        sd = os.path.join(args.out_dir, sys_name); os.makedirs(sd, exist_ok=True)
        np.savez_compressed(os.path.join(sd, "affinity_embeddings.npz"),
                            ids=np.array(ids), target=np.array([sys_name] * len(ids)),
                            pair_mean=PM, head_ens1=H1, head_ens2=H2, head_mean=(H1 + H2) / 2)
        print(f"  [{sys_name}] wrote {len(ids)} embeddings")
        all_ids += ids; all_pm.append(PM); all_h1.append(H1); all_h2.append(H2)

    if not all_ids:
        print("No embeddings produced."); return
    PM = np.concatenate(all_pm); H1 = np.concatenate(all_h1); H2 = np.concatenate(all_h2)
    ids = np.array(all_ids)
    tgt = np.array([i.split("::", 1)[0] for i in all_ids])
    combined = os.path.join(args.out_dir, "affinity_embeddings.npz")
    np.savez_compressed(combined, ids=ids, target=tgt,
                        pair_mean=PM, head_ens1=H1, head_ens2=H2, head_mean=(H1 + H2) / 2)
    with open(os.path.join(args.out_dir, "index.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(["row", "system", "input_id"])
        for i, idd in enumerate(ids):
            sysn, pid = str(idd).split("::", 1)
            w.writerow([i, sysn, pid])
    print(f"\nDONE n={len(ids)} pair_mean{PM.shape} head{H1.shape} -> {combined}")

    if args.validate_against:
        validate_against(args.validate_against, list(ids), list(PM), list(H1), list(H2),
                         args.cos_threshold)


if __name__ == "__main__":
    main()
