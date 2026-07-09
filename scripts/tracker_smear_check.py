"""How much of the real z_r_resid (~180) is physical measurement smearing vs the
trajectory itself? The source tracker_hits table has BOTH the measured (x,y,z) and the
true (true_x,y,z) hit positions, so we can decompose:

  z_r_resid(measured)  = what our tracker is trained on (includes smearing)
  z_r_resid(true)      = the trajectory alone (no smearing)
  per-hit smearing     = |measured - true|, split into transverse and z

If z_r_resid(true) << z_r_resid(measured), the "coherence scatter" is mostly physical
smearing that gen is SUPPOSED to have -> the model's excess (gen 216 vs real 180 = 36)
is the real target, and modelling the true trajectory + calibrated smearing (Option A)
is the clean decomposition. If z_r_resid(true) ~ z_r_resid(measured), the residual is
the trajectory (curvature / loopers), not smearing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard


def zr_resid(r, z):
    A = np.vstack([r, np.ones_like(r)]).T
    return float(np.sqrt(np.mean((z - A @ np.linalg.lstsq(A, z, rcond=None)[0]) ** 2)))


def r_mono(r):
    return float(np.mean(np.diff(np.sort(r)) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_tracks", type=int, default=20000)
    args = ap.parse_args()
    t = load_shard("tracker_hits", args.shard)
    col = {n: t[n].to_numpy(zero_copy_only=False) for n in
           ["event_id", "particle_id", "x", "y", "z", "true_x", "true_y", "true_z"]}
    ev, pid = col["event_id"].astype(np.int64), col["particle_id"].astype(np.int64)
    key = ev * (pid.max() + 1) + pid
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    bounds = np.concatenate([[0], np.where(np.diff(key_s))[0] + 1, [len(key_s)]])
    rng = np.random.default_rng(0)
    starts = rng.permutation(len(bounds) - 1)[:args.n_tracks]

    zrm, zrt, rmm, rmt = [], [], [], []
    smear_t, smear_z = [], []          # transverse and z smearing per hit
    for gi in starts:
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) < 3:
            continue
        x, y, z = col["x"][idx], col["y"][idx], col["z"][idx]
        tx, ty, tz = col["true_x"][idx], col["true_y"][idx], col["true_z"][idx]
        rm, rt = np.hypot(x, y), np.hypot(tx, ty)
        om, ot = np.argsort(rm), np.argsort(rt)
        zrm.append(zr_resid(rm[om], z[om])); zrt.append(zr_resid(rt[ot], tz[ot]))
        rmm.append(r_mono(rm)); rmt.append(r_mono(rt))
        smear_t.append(np.hypot(x - tx, y - ty)); smear_z.append(np.abs(z - tz))
    zrm, zrt = np.array(zrm), np.array(zrt)
    st, sz = np.concatenate(smear_t), np.concatenate(smear_z)
    print("=" * 60); print(f"TRACKER SMEARING CHECK (shard {args.shard}, {len(zrm)} tracks)"); print("=" * 60)
    print(f"z_r_resid  MEASURED  mean={zrm.mean():8.2f}  median={np.median(zrm):8.2f}")
    print(f"z_r_resid  TRUE      mean={zrt.mean():8.2f}  median={np.median(zrt):8.2f}")
    print(f"           -> true/measured ratio (mean) = {zrt.mean()/max(zrm.mean(),1e-9):.3f}")
    print(f"r_mono     MEASURED  mean={np.mean(rmm):.4f}   TRUE mean={np.mean(rmt):.4f}")
    print(f"per-hit smearing  transverse  mean={st.mean():.3f}  median={np.median(st):.3f}  p99={np.percentile(st,99):.2f}")
    print(f"per-hit smearing  z           mean={sz.mean():.3f}  median={np.median(sz):.3f}  p99={np.percentile(sz,99):.2f}")
    print(f"(units = position units of the source, same as r/z; gen z_r_resid ~216, real ~180)")


if __name__ == "__main__":
    main()
