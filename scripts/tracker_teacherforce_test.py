"""Exposure-bias diagnostic for the v3 surface tracker: per-step mean module-r for
  real  vs  free-running (own history)  vs  teacher-forced (true history),
on the SAME pion tracks. If teacher-forced is flat (≈real) but free-running climbs, the
per-step drift is exposure bias (→ helix/scheduled-sampling is the right fix). If teacher-
forced ALSO climbs, the conditional itself is mislearned."""
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
from genpu.models.tracker_module_model import TrackerModuleModel
from genpu.module_geometry import ModuleGeometry

DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"


def per_step_mean(vals_by_track, step_cap=16):
    """vals_by_track: list of 1D arrays (r per hit, inner->outer). -> mean per step index."""
    out = np.full(step_cap, np.nan)
    for k in range(step_cap):
        vk = [t[k] for t in vals_by_track if len(t) > k]
        if len(vk) > 30:
            out[k] = np.mean(vk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{DATA}/checkpoints/tracker/surface_pion_v3/checkpoint_060000.pt")
    ap.add_argument("--module_geometry", default=f"{DATA}/module_geometry.npz")
    ap.add_argument("--slice", default=f"{DATA}/tracker_slice/surface_pion.npz")
    ap.add_argument("--n_tracks", type=int, default=20000)
    ap.add_argument("--min_hits", type=int, default=4, help="only multi-hit tracks (drift needs length)")
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/tracker/v3_teacherforce_drift.png")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)

    d = np.load(args.slice)
    norm = {"cont_mean": d["cont_mean"], "cont_std": d["cont_std"]}
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    mg = ModuleGeometry(args.module_geometry)
    r_center = np.hypot(mg.means[:, 0], mg.means[:, 1]).astype(np.float32)   # module-center r
    r_centerT = torch.as_tensor(r_center, device=dev)

    nph = np.diff(off)
    sel = np.where(nph >= args.min_hits)[0]
    rng = np.random.default_rng(0); rng.shuffle(sel)
    sel = sel[:args.n_tracks]
    M = args.max_hits

    model = TrackerModuleModel(norm, module_geometry_path=args.module_geometry, max_hits=M).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()

    real_r, tf_r, free_r = [], [], []
    for s in range(0, len(sel), args.batch):
        idx = sel[s:s + args.batch]; B = len(idx)
        n_hits = np.minimum(nph[idx], M).astype(np.int64)
        module = np.zeros((B, M), np.int64); cont4 = np.zeros((B, M, 4), np.float32)
        mask = np.zeros((B, M), bool)
        for j, i in enumerate(idx):
            a = off[i]; nh = n_hits[j]
            module[j, :nh] = hits[a:a + nh, 0].astype(np.int64)
            cont4[j, :nh] = hits[a:a + nh, 1:5]
            mask[j, :nh] = True
        moduleT = torch.as_tensor(module, device=dev)
        cont4T = torch.as_tensor(cont4, device=dev)
        maskT = torch.as_tensor(mask, device=dev)
        nhT = torch.as_tensor(n_hits, device=dev)
        cS = torch.as_tensor((cont[idx] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
        pg = torch.as_tensor(pdg[idx], dtype=torch.long, device=dev)
        with torch.no_grad():
            ce = model.cond_embed(cS, pg)
            tf_mod = model.tracker.teacher_forced_modules(moduleT, cont4T, ce, maskT)   # (B,M)
            _, free_mod = model.tracker.generate(ce, nhT)                                # (B,max_n)
        rc_real = r_centerT[moduleT].cpu().numpy()
        rc_tf = r_centerT[tf_mod].cpu().numpy()
        rc_free = r_centerT[free_mod].cpu().numpy()
        for j in range(B):
            k = int(n_hits[j])
            real_r.append(rc_real[j, :k]); tf_r.append(rc_tf[j, :k]); free_r.append(rc_free[j, :k])

    steps = np.arange(16)
    R = per_step_mean(real_r); T = per_step_mean(tf_r); F = per_step_mean(free_r)
    print(f"{'step':>4} {'real':>8} {'teacher':>8} {'free':>8}")
    for k in steps:
        if not np.isnan(R[k]):
            print(f"{k:>4} {R[k]:>8.1f} {T[k]:>8.1f} {F[k]:>8.1f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps, R, "o-", color="k", lw=2, label="real (true modules)")
    ax.plot(steps, T, "s-", color="tab:green", lw=2, label="teacher-forced (true history)")
    ax.plot(steps, F, "^-", color="tab:blue", lw=2, label="free-running (own history)")
    ax.set_xlabel("hit index in track (inner→outer)"); ax.set_ylabel("mean module-center r [mm]")
    ax.set_title(f"v3 per-step drift: exposure-bias test ({len(sel)} tracks, ≥{args.min_hits} hits)\n"
                 "teacher-forced ≈ real; free-running gap = residual drift")
    ax.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
