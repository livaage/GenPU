"""Pre-build test for the FIXED-initial-momentum helix anchor. Read-only, login-node.

For real charged tracks: (1) fit the transverse trajectory circle from the hits and measure the
residual (scatter around the circle = the 'deviation' the model would predict — must be SMALL);
(2) check the fitted curvature is predicted by the INITIAL pT via a single global B (fixed-momentum
assumption — energy loss would break this); (3) z-vs-r line residual + slope vs sinh(eta).
If circle residuals are small AND implied-B clusters tightly, the fixed helix is a good reference.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"
PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)


def fit_circle(x, y):
    """Algebraic (Kasa) circle fit. Returns center (cx,cy), R, rms radial residual [mm]."""
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    R = np.sqrt(max(cx ** 2 + cy ** 2 - F, 1e-6))
    resid = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - R) ** 2))
    return cx, cy, R, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--n_tracks", type=int, default=20000)
    ap.add_argument("--min_hits", type=int, default=4)
    args = ap.parse_args()

    d = np.load(f"{DATA}/preprocessed/shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    nph = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (nph >= args.min_hits))[0]
    rng = np.random.default_rng(0); rng.shuffle(sel); sel = sel[:args.n_tracks]

    circ_res, z_res, implied_B, R_fit_l, pT_l, slope_err = [], [], [], [], [], []
    res_by_r = {lo: [] for lo in (0, 200, 400, 600, 800)}   # circle resid vs radius (energy-loss probe)
    for i in sel:
        a, b = off[i], off[i + 1]
        r, phi, z = th[a:b, TH_R], th[a:b, TH_PHI], th[a:b, TH_Z]
        x, y = r * np.cos(phi), r * np.sin(phi)
        cx, cy, R, cres = fit_circle(x, y)
        pT = np.exp(pf[i, PF_LOGPT]); q = pf[i, PF_CHARGE]
        if abs(q) < 0.5 or R < 1e-3:
            continue
        circ_res.append(cres)
        R_fit_l.append(R); pT_l.append(pT)
        # implied B from R = pT/(0.3 |q| B)  ->  B = pT/(0.3 |q| R)   [GeV, T, m];  R in mm -> /1000
        implied_B.append(pT / (0.3 * abs(q) * (R / 1000.0)))
        # z vs r line
        A = np.stack([r, np.ones_like(r)], 1)
        m, c = np.linalg.lstsq(A, z, rcond=None)[0]
        z_res.append(np.sqrt(np.mean((z - (m * r + c)) ** 2)))
        slope_err.append(m - np.sinh(pf[i, PF_ETA]))
        for lo in res_by_r:
            mask = (r >= lo) & (r < lo + 200)
            if mask.sum() >= 1:
                res_by_r[lo].append(cres)

    circ_res = np.array(circ_res); z_res = np.array(z_res)
    implied_B = np.array(implied_B); R_fit = np.array(R_fit_l); pT = np.array(pT_l)
    print(f"tracks used: {len(circ_res)}  (pdg={args.pdg_class}, >={args.min_hits} hits)")
    print("=" * 64)
    print("(1) TRANSVERSE circle-fit residual [mm]  (the model's 'deviation' scale)")
    p = np.percentile(circ_res, [50, 90, 99])
    print(f"    median {p[0]:.2f}   p90 {p[1]:.2f}   p99 {p[2]:.2f}")
    print("(2) z-vs-r line residual [mm]  +  slope vs sinh(eta)")
    pz = np.percentile(z_res, [50, 90, 99])
    print(f"    z_resid median {pz[0]:.2f}  p90 {pz[1]:.2f}   slope_err median {np.median(slope_err):+.3f}")
    print("(3) IMPLIED B [T] from fitted R and initial pT  (tight cluster => fixed-momentum holds)")
    # restrict to well-curved tracks (R < 5 m) where curvature is well-measured
    wc = R_fit < 5000
    pb = np.percentile(implied_B[wc], [10, 25, 50, 75, 90])
    print(f"    well-curved tracks: {wc.sum()}  implied B: p10 {pb[0]:.2f}  p25 {pb[1]:.2f}  "
          f"median {pb[2]:.2f}  p75 {pb[3]:.2f}  p90 {pb[4]:.2f}")
    print(f"    IQR/median = {(pb[3]-pb[1])/pb[2]:.2f}  (small => consistent single B)")
    print("(4) circle residual vs hit radius (rising => energy-loss curvature the fixed helix misses)")
    for lo in sorted(res_by_r):
        v = np.array(res_by_r[lo])
        if len(v) > 50:
            print(f"    r in [{lo},{lo+200}) mm: median circ_resid {np.median(v):.2f} mm  (n={len(v)})")


if __name__ == "__main__":
    main()
