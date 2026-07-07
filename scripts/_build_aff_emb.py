import numpy as np, glob, os, json, torch, csv, sys
root="/work/jwang/boltz2/peptides/_output"
outdir="/work/jwang/boltz2/peptides/_affinity_embeddings"
os.makedirs(outdir, exist_ok=True)
targets=["bh3__Bcl-xL","bh3__Bfl-1","bh3__Mcl-1","p53__MDM2","p53__MDMX"]

# load affinity_out_mlp (both ensemble members) from checkpoint
ck=torch.load("/home/juw79/.boltz/boltz2_aff.ckpt",map_location="cpu",weights_only=False)
sd=ck.get("state_dict",ck)
def mlp(mod):
    p=f"affinity_module{mod}.affinity_heads.affinity_out_mlp."
    return [(sd[p+"0.weight"].float().numpy(), sd[p+"0.bias"].float().numpy()),
            (sd[p+"2.weight"].float().numpy(), sd[p+"2.bias"].float().numpy())]
M1,M2=mlp(1),mlp(2)
def apply_mlp(x,M):
    (W0,b0),(W2,b2)=M
    h=np.maximum(x@W0.T+b0,0.0)        # Linear -> ReLU
    return h@W2.T+b2                    # Linear
def chainB_tokens(stpath):
    s=np.load(stpath,allow_pickle=True); ch=s["chains"]
    n=int(ch['res_num'].sum())
    rngs={str(c['name']):(int(c['res_idx']),int(c['res_num'])) for c in ch}
    return n,rngs

ids=[]; tgs=[]; PM=[]; H1=[]; H2=[]
done=0
for t in targets:
    embs=sorted(glob.glob(f"{root}/{t}/**/embeddings_*.npz",recursive=True))
    for f in embs:
        pid=os.path.basename(f).replace("embeddings_","").replace(".npz","")
        sts=glob.glob(f"{root}/{t}/**/structures/{pid}.npz",recursive=True)
        if not sts:
            print("WARN no structure for",t,pid); continue
        n,rngs=chainB_tokens(sts[0])
        z=np.load(f)["z"][0]            # (N,N,128)
        N=z.shape[0]
        a0,an=rngs["A"]; b0,bn=rngs["B"]
        lig=np.zeros(N,bool); lig[b0:b0+bn]=True
        rec=np.zeros(N,bool); rec[a0:a0+an]=True
        L=lig[:,None]; R=rec[:,None]
        mask=(L&R.T)|(R&L.T)|(L&L.T)
        np.fill_diagonal(mask,False)
        m=mask.astype(np.float32)
        pm=(z*m[:,:,None]).sum((0,1))/(m.sum()+1e-7)   # (128,)
        PM.append(pm); H1.append(apply_mlp(pm,M1)); H2.append(apply_mlp(pm,M2))
        ids.append(f"{t}::{pid}"); tgs.append(t)
        done+=1
        if done%200==0: print(f"  {done} done",flush=True)

PM=np.array(PM,np.float32); H1=np.array(H1,np.float32); H2=np.array(H2,np.float32)
ids=np.array(ids); tgs=np.array(tgs)
np.savez_compressed(f"{outdir}/affinity_embeddings.npz",
    ids=ids, target=tgs, pair_mean=PM, head_ens1=H1, head_ens2=H2, head_mean=(H1+H2)/2)
with open(f"{outdir}/index.tsv","w",newline="") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["row","target","peptide_id"])
    for i,(tg,idd) in enumerate(zip(tgs,ids)): w.writerow([i,tg,idd.split("::",1)[1]])
print(f"DONE n={len(ids)} pair_mean{PM.shape} head{H1.shape} -> {outdir}/affinity_embeddings.npz")
