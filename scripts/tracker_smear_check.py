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
from genpu.data import load_shard, explode_list_columns


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
    t = load_shard("tracker_hits", args.shard)          # one row per event, list columns
    zrm, zrt, rmm, rmt = [], [], [], []
    smear_t, smear_z = [], []                            # transverse and z smearing per hit
    for eidx in range(t.num_rows):
        ev = explode_list_columns(t, eidx)
        pid = ev["particle_id"].astype(np.int64)
        x, y, z = ev["x"], ev["y"], ev["z"]
        tx, ty, tz = ev["true_x"], ev["true_y"], ev["true_z"]
        for p in np.unique(pid):
            m = pid == p
            if m.sum() < 3:
                continue
            xm, ym, zm = x[m], y[m], z[m]
            xt, yt, zt = tx[m], ty[m], tz[m]
            rm, rt = np.hypot(xm, ym), np.hypot(xt, yt)
            om, ot = np.argsort(rm), np.argsort(rt)
            zrm.append(zr_resid(rm[om], zm[om])); zrt.append(zr_resid(rt[ot], zt[ot]))
            rmm.append(r_mono(rm)); rmt.append(r_mono(rt))
            smear_t.append(np.hypot(xm - xt, ym - yt)); smear_z.append(np.abs(zm - zt))
        if len(zrm) >= args.n_tracks:
            break
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
