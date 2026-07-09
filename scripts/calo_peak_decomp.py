"""Decompose the d_eta/d_phi peak: is the residual tip from the DELTA flow or the
per-shower CORE (global mixture)? d_eta = core + delta. Report the central-bin
concentration (|x|<w) for real vs gen, separately for core, delta, and their sum."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.flow.calo_flow import CaloFlow


def central(x, w):
    return float(np.mean(np.abs(x) < w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon_core.npz")
    ap.add_argument("--n_showers", type=int, default=20000)
    ap.add_argument("--w", type=float, default=0.034)   # half-width of the central bin
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(1)

    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std", "glob_mean", "glob_std", "pts_mean", "pts_std"]}
    cont, pdg, glob, pts, off = d["cont"], d["pdg"], d["glob"], d["points_flat"], d["offsets"]
    val = rng.permutation(cont.shape[0])[:args.n_showers]
    model = CaloFlow(norm).to(dev); model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()

    # REAL: core per shower (glob[:,2:4]); delta per cell (pts[:,:2])
    t_n = np.array([off[i + 1] - off[i] for i in val])
    r_core = glob[val, 2:4]
    r_delta = np.concatenate([pts[off[i]:off[i + 1], :2] for i in val])

    # GEN
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    with torch.no_grad():
        ce = model.cond_embed(contS, pdgT); g_std = model.glob.sample(ce)
        g = model.unstd_glob(g_std).cpu().numpy()
    g_core = g[:, 2:4]
    g_n = np.clip(np.round(np.exp(g[:, 1])).astype(int), 1, int(t_n.max()) + 5)
    rep = torch.as_tensor(np.repeat(np.arange(len(val)), g_n), device=dev)
    with torch.no_grad():
        g_delta = model.unstd_pos(model.points.sample(ce[rep], g_std[rep][:, :2], steps=100)).cpu().numpy()

    r_sum = r_delta + np.repeat(r_core, t_n, axis=0)
    g_sum = g_delta + np.repeat(g_core, g_n, axis=0)

    print("=" * 64); print("CALO PEAK DECOMPOSITION —", Path(args.ckpt).name, f"(|x|<{args.w})"); print("=" * 64)
    print(f"{'component':14s} {'dim':4s} {'real':>8s} {'gen':>8s} {'gen/real':>9s}")
    for label, rv, gv in [("core", r_core, g_core), ("delta", r_delta, g_delta), ("sum(d_eta)", r_sum, g_sum)]:
        for j, dn in enumerate(["eta", "phi"]):
            cr, cg = central(rv[:, j], args.w), central(gv[:, j], args.w)
            print(f"  {label:12s} {dn:4s} {cr:8.3f} {cg:8.3f} {cg/max(cr,1e-9):9.2f}")
    # also std, to see which is broader
    print("-- std --")
    for label, rv, gv in [("core", r_core, g_core), ("delta", r_delta, g_delta)]:
        for j, dn in enumerate(["eta", "phi"]):
            print(f"  {label:12s} {dn:4s} real_std={rv[:,j].std():.4f} gen_std={gv[:,j].std():.4f}")


if __name__ == "__main__":
    main()
