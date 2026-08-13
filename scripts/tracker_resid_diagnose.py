"""Per-hit residual diagnosis: WHERE does the AR over-reach?
Inverts physical -> per-layer-standardized residual (resid = (phys - LAYER_MEAN)/LAYER_STD)
for BOTH real and generated hits, then compares:
  (1) residual distributions (r_resid, z_resid): percentiles + tail/edge fractions
  (2) bin range / clipping: frac of real residuals clipped at |resid|>3 (info lost at train),
      vs frac of GEN residuals piled in the top edge bin -> is the head over-weighting edges?
  (3) per-step (hit index in track) residual mean/std, real vs gen -> exposure-bias DRIFT test
Generation matches the honest gate (count head n_hits, temp=1, use_vertex).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from genpu.models.count_head import CountHead
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)
LM = LAYER_MEANS.astype(np.float32); LS = LAYER_STDS.astype(np.float32)


def resid_of(layer, r, z):
    lc = np.clip(np.round(layer).astype(int), 0, N_LAYERS - 1)
    return (r - LM[lc, 0]) / LS[lc, 0], (z - LM[lc, 2]) / LS[lc, 2]


def summ(name, x):
    p = np.percentile(x, [50, 90, 99, 99.9])
    return (f"  {name:10s} mean {x.mean():+6.3f} std {x.std():5.3f}  "
            f"p50 {p[0]:+.2f} p90 {p[1]:+.2f} p99 {p[2]:+.2f} p99.9 {p[3]:+.2f}  "
            f"frac>2.5 {np.mean(x>2.5):.4f}  frac>=2.9(edge) {np.mean(x>=2.9):.4f}  "
            f"frac|.|>3(clip) {np.mean(np.abs(x)>3):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--count_ckpt", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    ap.add_argument("--max_hits", type=int, default=32); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--use_vertex", action="store_true"); ap.add_argument("--count_no_d0", action="store_true")
    ap.add_argument("--use_mom_feat", action="store_true")
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/tracker")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)

    dsl = np.load(args.slice); norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModel(norm, use_vertex=args.use_vertex, use_mom_feat=args.use_mom_feat).to(dev)
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
    phi = pf[sel, PF_PHI]; d0 = (aux[sel, AUX_VX] * np.sin(phi) - aux[sel, AUX_VY] * np.cos(phi)).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64)
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev); d0T = torch.as_tensor(d0, device=dev)
    vtxT = torch.as_tensor(cont[:, [5, 6]], dtype=torch.float32, device=dev)
    helixT = (TrackerModel.helix_params_from_cont(torch.as_tensor(cont, dtype=torch.float32, device=dev))
              if args.use_mom_feat else None)

    # REAL residuals + step index (hits already sorted inner->outer in stage2)
    r_rr, r_zr, r_step = [], [], []
    for i in sel:
        a, b = off[i], off[i + 1]
        rr, zr = resid_of(th[a:b, TH_LAYER], th[a:b, TH_R], th[a:b, TH_Z])
        r_rr.append(rr); r_zr.append(zr); r_step.append(np.arange(b - a))
    r_rr = np.concatenate(r_rr); r_zr = np.concatenate(r_zr); r_step = np.concatenate(r_step)

    # GEN (honest count, temp=1)
    with torch.no_grad():
        n_gen = ch.sample(contS, pdgT, d0T).clamp(1, args.max_hits)
    n_gen_np = n_gen.cpu().numpy()
    g_rr, g_zr, g_step = [], [], []
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            vtx = vtxT[s:e] if args.use_vertex else None
            hlx = helixT[s:e] if args.use_mom_feat else None
            hits, layers = model.tracker.generate(ce, n_gen[s:e], vertex_pos=vtx, helix_params=hlx)
        hits = hits.cpu().numpy(); layers = layers.cpu().numpy()
        for j in range(e - s):
            k = int(n_gen_np[s + j])
            rr, zr = resid_of(layers[j, :k], hits[j, :k, TH_R - 1], hits[j, :k, TH_Z - 1])
            g_rr.append(rr); g_zr.append(zr); g_step.append(np.arange(k))
    g_rr = np.concatenate(g_rr); g_zr = np.concatenate(g_zr); g_step = np.concatenate(g_step)

    print("=" * 92)
    print(f"PER-HIT RESIDUAL DIAGNOSIS  pdg=all  real hits={len(r_rr)}  gen hits={len(g_rr)}")
    print("=" * 92)
    print("(1) r_resid distribution");  print("REAL" + summ("r", r_rr)); print("GEN " + summ("r", g_rr))
    print("(1) z_resid distribution");  print("REAL" + summ("z", r_zr)); print("GEN " + summ("z", g_zr))
    print("\n(2) BIN RANGE / CLIPPING")
    print(f"  real r_resid clipped (|.|>3): {np.mean(np.abs(r_rr)>3):.4f}   "
          f"gen at top edge (>=2.9): {np.mean(g_rr>=2.9):.4f} vs real {np.mean(r_rr>=2.9):.4f}  "
          f"-> ratio {np.mean(g_rr>=2.9)/max(np.mean(r_rr>=2.9),1e-6):.1f}x")
    print(f"  real z_resid clipped (|.|>3): {np.mean(np.abs(r_zr)>3):.4f}   "
          f"gen at top edge (>=2.9): {np.mean(g_zr>=2.9):.4f} vs real {np.mean(r_zr>=2.9):.4f}  "
          f"-> ratio {np.mean(g_zr>=2.9)/max(np.mean(r_zr>=2.9),1e-6):.1f}x")
    print("\n(3) PER-STEP DRIFT (r_resid mean/std by hit index; exposure-bias test)")
    print(f"  {'step':>4s} {'realN':>8s} {'genN':>8s} {'real_mean':>9s} {'gen_mean':>9s} {'real_std':>9s} {'gen_std':>9s}")
    for st in range(0, 12):
        rm = r_step == st; gm = g_step == st
        if rm.sum() < 50 or gm.sum() < 50:
            continue
        print(f"  {st:>4d} {rm.sum():>8d} {gm.sum():>8d} {r_rr[rm].mean():>9.3f} {g_rr[gm].mean():>9.3f} "
              f"{r_rr[rm].std():>9.3f} {g_rr[gm].std():>9.3f}")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.linspace(-3.2, 3.2, 130)
    ax[0].hist(r_rr, bins=bins, histtype="step", lw=2, density=True, label="real"); ax[0].hist(g_rr, bins=bins, histtype="step", lw=2, density=True, label="gen")
    ax[0].set_yscale("log"); ax[0].set_xlabel("r_resid"); ax[0].set_title("r residual (log y)"); ax[0].legend()
    ax[1].hist(r_zr, bins=bins, histtype="step", lw=2, density=True, label="real"); ax[1].hist(g_zr, bins=bins, histtype="step", lw=2, density=True, label="gen")
    ax[1].set_yscale("log"); ax[1].set_xlabel("z_resid"); ax[1].set_title("z residual (log y)"); ax[1].legend()
    steps = np.arange(12)
    rmean = [r_rr[r_step == s].mean() if (r_step == s).sum() > 50 else np.nan for s in steps]
    gmean = [g_rr[g_step == s].mean() if (g_step == s).sum() > 50 else np.nan for s in steps]
    ax[2].plot(steps, rmean, "o-", label="real"); ax[2].plot(steps, gmean, "s-", label="gen")
    ax[2].set_xlabel("hit index in track"); ax[2].set_ylabel("r_resid mean"); ax[2].set_title("drift vs step"); ax[2].legend()
    plt.tight_layout(); fig.savefig(Path(args.outdir) / "resid_diag.png", dpi=100)
    print(f"\nwrote {Path(args.outdir)/'resid_diag.png'}")


if __name__ == "__main__":
    main()
