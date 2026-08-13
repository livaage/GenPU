"""Step-0 (innermost hit) check: is the first generated hit genuinely at smaller PHYSICAL r,
or does it just land in a slightly-too-outer LAYER (making the within-layer residual read negative)?
Compares, for the innermost hit of each track (and steps 1,2 for context): physical r, physical |z|,
and the layer-class choice, gen vs real. Generation matches the honest gate (temp=1, use_vertex)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from genpu.models.count_head import CountHead
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)
LM = LAYER_MEANS.astype(np.float32); LS = LAYER_STDS.astype(np.float32)


def report(tag, step, lay, r, z):
    lc = np.clip(np.round(lay).astype(int), 0, N_LAYERS - 1)
    rres = (r - LM[lc, 0]) / LS[lc, 0]
    pr = np.percentile(r, [10, 50, 90])
    # top layers by frequency
    u, c = np.unique(lc, return_counts=True); top = u[np.argsort(c)[::-1][:5]]
    topf = {int(t): round(float((lc == t).mean()), 3) for t in top}
    print(f"  {tag:5s} step{step}: n={len(r):8d}  r_phys[mean {r.mean():7.1f} med {pr[1]:7.1f} p10 {pr[0]:7.1f} p90 {pr[2]:7.1f}]  "
          f"|z|_mean {np.abs(z).mean():7.1f}  layer[mean {lc.mean():5.2f} med {np.median(lc):4.0f}]  r_resid_mean {rres.mean():+.3f}")
    print(f"         top-5 layers (frac): {topf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--count_ckpt", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=list(range(15)))
    ap.add_argument("--max_hits", type=int, default=32); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--use_vertex", action="store_true"); ap.add_argument("--count_no_d0", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)

    dsl = np.load(args.slice); norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModel(norm, use_vertex=args.use_vertex).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    ch = CountHead(use_d0=not args.count_no_d0).to(dev)
    ch.load_state_dict(torch.load(args.count_ckpt, map_location=dev)["model"]); ch.eval()

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    ntrk = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ntrk >= 1))[0]
    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY]); logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], axis=1).astype(np.float32)
    phi = pf[sel, PF_PHI]; d0 = (aux[sel, AUX_VX]*np.sin(phi) - aux[sel, AUX_VY]*np.cos(phi)).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64)
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev); d0T = torch.as_tensor(d0, device=dev)
    vtxT = torch.as_tensor(cont[:, [5, 6]], dtype=torch.float32, device=dev)

    # REAL step-0/1/2
    R = {s: {"lay": [], "r": [], "z": []} for s in (0, 1, 2)}
    for i in sel:
        a, b = off[i], off[i + 1]; n = b - a
        for s in (0, 1, 2):
            if s < n:
                R[s]["lay"].append(th[a + s, TH_LAYER]); R[s]["r"].append(th[a + s, TH_R]); R[s]["z"].append(th[a + s, TH_Z])

    with torch.no_grad():
        n_gen = ch.sample(contS, pdgT, d0T).clamp(1, args.max_hits)
    n_gen_np = n_gen.cpu().numpy()
    G = {s: {"lay": [], "r": [], "z": []} for s in (0, 1, 2)}
    for s0 in range(0, len(sel), args.batch):
        e = min(s0 + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s0:e], pdgT[s0:e])
            vtx = vtxT[s0:e] if args.use_vertex else None
            hits, layers = model.tracker.generate(ce, n_gen[s0:e], vertex_pos=vtx)
        hits = hits.cpu().numpy(); layers = layers.cpu().numpy()
        for j in range(e - s0):
            k = int(n_gen_np[s0 + j])
            for s in (0, 1, 2):
                if s < k:
                    G[s]["lay"].append(layers[j, s]); G[s]["r"].append(hits[j, s, TH_R - 1]); G[s]["z"].append(hits[j, s, TH_Z - 1])

    print("=" * 100)
    print("STEP-0 CHECK: physical r + layer choice, gen vs real (innermost hit = smaller absolute r?)")
    print("=" * 100)
    for s in (0, 1, 2):
        report("REAL", s, np.array(R[s]["lay"]), np.array(R[s]["r"]), np.array(R[s]["z"]))
        report("GEN", s, np.array(G[s]["lay"]), np.array(G[s]["r"]), np.array(G[s]["z"]))
        print()


if __name__ == "__main__":
    main()
