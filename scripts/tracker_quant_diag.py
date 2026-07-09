"""Is the tracker's under-filled COHERENCE peak (r_mono=1, z_r_resid=0, phi_r_resid=0)
caused by the residual BINNING, or by the AR's per-hit prediction?

Test: take REAL tracks and push their per-layer residuals through the model's exact
128-bin quantization (digitize -> bin center -> dequantize), then recompute the
coherence features. If quantized-real collapses toward gen, the BINNING resolution is
the bottleneck (the residual scheme is the problem). If quantized-real stays sharp like
real, the binning is innocent and it's the AR's per-hit independence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS

N_BINS, LO, HI = 128, -3.0, 3.0
CENTERS = (np.linspace(LO, HI, N_BINS + 1)[:-1] + np.linspace(LO, HI, N_BINS + 1)[1:]) / 2


def quantize(resid):
    """residual -> model bin center (matches tracker_ar._digitize + _spatial_centers)."""
    clamped = np.clip(resid, LO, HI)
    idx = np.round((clamped - LO) / (HI - LO) * (N_BINS - 1)).astype(int).clip(0, N_BINS - 1)
    return CENTERS[idx]


def track_feats(r, phi, z):
    n = len(r)
    if n < 3:
        return None
    order = np.argsort(r); r, phi, z = r[order], np.unwrap(phi[order]), z[order]
    r_mono = float(np.mean(np.diff(r) > 0)); dr_std = float(np.diff(r).std())
    A = np.vstack([r, np.ones_like(r)]).T
    z_res = float(np.sqrt(np.mean((z - A @ np.linalg.lstsq(A, z, rcond=None)[0]) ** 2)))
    phi_res = float(np.sqrt(np.mean((phi - A @ np.linalg.lstsq(A, phi, rcond=None)[0]) ** 2)))
    return [n, r.mean(), r_mono, z_res, phi_res, dr_std]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--n_tracks", type=int, default=12000)
    ap.add_argument("--max_hits", type=int, default=32)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    d = np.load(args.slice)
    hits, off = d["hits"], d["offsets"]
    S = len(off) - 1
    val = rng.permutation(S)[:args.n_tracks]
    LM, LS = LAYER_MEANS, LAYER_STDS

    names = ["n_hits", "r_mean", "r_mono", "z_r_resid", "phi_r_resid", "dr_std"]
    Xr, Xq = [], []
    for gi in val:
        a, b = off[gi], min(off[gi] + args.max_hits, off[gi + 1])
        lc = hits[a:b, 0].astype(int).clip(0, N_LAYERS - 1)
        res = hits[a:b, 1:4]                                   # per-layer residuals (r,phi,z)
        # REAL physical
        rr = res[:, 0] * LS[lc, 0] + LM[lc, 0]
        pp = res[:, 1] * LS[lc, 1] + LM[lc, 1]
        zz = res[:, 2] * LS[lc, 2] + LM[lc, 2]
        f = track_feats(rr, pp, zz)
        if f: Xr.append(f)
        # QUANTIZED-real physical (residuals through the model bins)
        qr = quantize(res[:, 0]) * LS[lc, 0] + LM[lc, 0]
        qp = quantize(res[:, 1]) * LS[lc, 1] + LM[lc, 1]
        qz = quantize(res[:, 2]) * LS[lc, 2] + LM[lc, 2]
        f = track_feats(qr, qp, qz)
        if f: Xq.append(f)
    Xr, Xq = np.array(Xr), np.array(Xq)
    print("=" * 66); print("TRACKER BINNING-QUANTIZATION TEST (real vs quantized-real)"); print("=" * 66)
    print(f"{'feature':14s} {'real':>12s} {'quant-real':>12s} {'shift':>10s}   (gen ref, vertex model)")
    genref = {"r_mono": 0.876, "z_r_resid": 168.0, "phi_r_resid": 0.181, "dr_std": 31.4}
    for j, nm in enumerate(names):
        gr = f"{genref[nm]:.3f}" if nm in genref else "-"
        print(f"  {nm:12s} {Xr[:,j].mean():12.4f} {Xq[:,j].mean():12.4f} "
              f"{(Xq[:,j].mean()-Xr[:,j].mean()):10.4f}   gen~{gr}")


if __name__ == "__main__":
    main()
