"""Compare real vs v1 (layer-residual) vs v3 (surface-local) tracker generation.
Truth-count generation for both models (matched n_hits) so only spatial quality differs.
Panels: r, z, layer_class marginals + the per-step mean-r DRIFT curve (the money plot)."""
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
from genpu.models.tracker_module_model import TrackerModuleModel
from genpu.module_geometry import ModuleGeometry

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)
DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"


def cond_from_stage2(pf, aux, sel):
    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY])
    logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    return np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], axis=1).astype(np.float32)


def per_step_mean_r(r_list, step_list, kmax=16):
    r = np.concatenate(r_list); s = np.concatenate(step_list)
    return np.array([r[s == k].mean() if (s == k).sum() > 20 else np.nan for k in range(kmax)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1_ckpt", default=f"{DATA}/checkpoints/tracker/pion_baseline_current/checkpoint_060000.pt")
    ap.add_argument("--v3_ckpt", default=f"{DATA}/checkpoints/tracker/surface_pion_v3/checkpoint_060000.pt")
    ap.add_argument("--module_geometry", default=f"{DATA}/module_geometry.npz")
    ap.add_argument("--v1_slice", default=f"{DATA}/tracker_slice/pion.npz")
    ap.add_argument("--v3_slice", default=f"{DATA}/tracker_slice/surface_pion.npz")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_particles", type=int, default=30000)
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/tracker/v3_vs_v1_compare.png")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)

    d = np.load(Path(f"{DATA}/preprocessed") / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    ntrk = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], [3, 4]) & (ntrk >= 1))[0]
    if args.n_particles and len(sel) > args.n_particles:
        sel = sel[:args.n_particles]
    cont = cond_from_stage2(pf, aux, sel)
    pdg = pf[sel, PF_PDG].astype(np.int64)
    n_true = np.clip(ntrk[sel], 1, args.max_hits).astype(np.int64)
    nT = torch.as_tensor(n_true, device=dev)

    # real hits (sorted inner->outer already in stage2)
    real_r, real_z, real_l, real_step = [], [], [], []
    for i in sel:
        a, b = off[i], off[i + 1]
        real_l.append(th[a:b, TH_LAYER]); real_r.append(th[a:b, TH_R]); real_z.append(th[a:b, TH_Z])
        real_step.append(np.arange(b - a))

    # --- v1 generation ---
    n1 = np.load(args.v1_slice); norm1 = {"cont_mean": n1["cont_mean"], "cont_std": n1["cont_std"]}
    m1 = TrackerModel(norm1).to(dev)
    m1.load_state_dict(torch.load(args.v1_ckpt, map_location=dev)["model"]); m1.eval()
    cS1 = torch.as_tensor((cont - norm1["cont_mean"]) / norm1["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    v1_r, v1_z, v1_l, v1_step = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(sel), args.batch):
            e = min(s + args.batch, len(sel))
            ce = m1.cond_embed(cS1[s:e], pdgT[s:e])
            hits, layers = m1.tracker.generate(ce, nT[s:e])
            hits = hits.cpu().numpy(); layers = layers.cpu().numpy()
            for j in range(e - s):
                k = int(n_true[s + j])
                v1_r.append(hits[j, :k, TH_R - 1]); v1_z.append(hits[j, :k, TH_Z - 1])
                v1_l.append(layers[j, :k]); v1_step.append(np.arange(k))

    # --- v3 generation ---
    n3 = np.load(args.v3_slice); norm3 = {"cont_mean": n3["cont_mean"], "cont_std": n3["cont_std"]}
    m3 = TrackerModuleModel(norm3, module_geometry_path=args.module_geometry, max_hits=args.max_hits).to(dev)
    m3.load_state_dict(torch.load(args.v3_ckpt, map_location=dev)["model"]); m3.eval()
    mg = ModuleGeometry(args.module_geometry); lom = torch.as_tensor(mg.layer_class, dtype=torch.long, device=dev)
    cS3 = torch.as_tensor((cont - norm3["cont_mean"]) / norm3["cont_std"], device=dev)
    v3_r, v3_z, v3_l, v3_step = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(sel), args.batch):
            e = min(s + args.batch, len(sel))
            ce = m3.cond_embed(cS3[s:e], pdgT[s:e])
            phys, gmod = m3.tracker.generate(ce, nT[s:e])
            layers = lom[gmod].cpu().numpy(); phys = phys.cpu().numpy()
            r = np.hypot(phys[:, :, 0], phys[:, :, 1]); z = phys[:, :, 2]
            for j in range(e - s):
                k = int(n_true[s + j])
                v3_r.append(r[j, :k]); v3_z.append(z[j, :k]); v3_l.append(layers[j, :k]); v3_step.append(np.arange(k))

    # --- plot ---
    RR = np.concatenate(real_r); RZ = np.concatenate(real_z); RL = np.concatenate(real_l)
    A1R = np.concatenate(v1_r); A1Z = np.concatenate(v1_z); A1L = np.concatenate(v1_l)
    A3R = np.concatenate(v3_r); A3Z = np.concatenate(v3_z); A3L = np.concatenate(v3_l)
    C = {"real": "k", "v1": "tab:red", "v3": "tab:blue"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    rb = np.linspace(0, 1100, 80)
    for nm, a in (("real", RR), ("v1", A1R), ("v3", A3R)):
        ax[0, 0].hist(a, bins=rb, histtype="step", lw=2, density=True, color=C[nm],
                      label=f"{nm} (mean {a.mean():.0f})")
    ax[0, 0].set_xlabel("hit r [mm]"); ax[0, 0].set_title("radius (v1 drifts outward)"); ax[0, 0].legend()

    zb = np.linspace(-3200, 3200, 80)
    for nm, a in (("real", RZ), ("v1", A1Z), ("v3", A3Z)):
        ax[0, 1].hist(a, bins=zb, histtype="step", lw=2, density=True, color=C[nm], label=nm)
    ax[0, 1].set_xlabel("hit z [mm]"); ax[0, 1].set_title("z"); ax[0, 1].legend()

    lb = np.arange(-0.5, 48.5, 1)
    for nm, a in (("real", RL), ("v1", A1L), ("v3", A3L)):
        ax[1, 0].hist(a, bins=lb, histtype="step", lw=2, density=True, color=C[nm], label=nm)
    ax[1, 0].set_xlabel("layer_class"); ax[1, 0].set_title("layer occupancy"); ax[1, 0].legend()

    steps = np.arange(16)
    for nm, rl, sl in (("real", real_r, real_step), ("v1", v1_r, v1_step), ("v3", v3_r, v3_step)):
        ax[1, 1].plot(steps, per_step_mean_r(rl, sl), "o-", color=C[nm], label=nm)
    ax[1, 1].set_xlabel("hit index in track (inner→outer)"); ax[1, 1].set_ylabel("mean r [mm]")
    ax[1, 1].set_title("PER-STEP DRIFT — mean r vs step (v1 climbs, v3 flat like real)"); ax[1, 1].legend()

    fig.suptitle(f"Tracker pions (truth count, shard {args.shard}, {len(sel)} particles): "
                 f"real vs v1 layer-residual vs v3 surface-local", fontsize=13)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    print(f"r_mean  real {RR.mean():.1f}  v1 {A1R.mean():.1f}  v3 {A3R.mean():.1f}")
    print(f"z_std   real {RZ.std():.1f}  v1 {A1Z.std():.1f}  v3 {A3Z.std():.1f}")


if __name__ == "__main__":
    main()
