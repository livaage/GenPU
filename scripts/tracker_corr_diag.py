"""Are track residuals CORRELATED along the track? I.e. does a helical start stay
helical and a messy start stay messy? Two measures, real vs gen:

  1. lag-1 autocorrelation of the (signed) z-vs-r and phi-vs-r line-fit residuals
     within each track. High -> residuals are a SMOOTH function of r (one coherent
     trajectory, deviations persist); ~0 -> per-hit independent noise.
  2. quality persistence: split each track in half, measure LOCAL roughness (line-fit
     RMS) of each half, correlate first-half vs second-half roughness across tracks.
     High -> a clean/messy start predicts a clean/messy end.

If real is strongly correlated and gen is weaker, the model injects per-hit scatter
that decorrelates the trajectory -> the exact reason the coherence spikes are under-filled.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS


def _fit_resid(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def lag1(x):
    x = x - x.mean()
    d = np.mean(x * x)
    return float(np.mean(x[:-1] * x[1:]) / d) if (len(x) >= 3 and d > 1e-12) else None


def track_measures(r, phi, z):
    """Return (lag1_zr, lag1_pr, first_rms, second_rms) for one track, or None."""
    if len(r) < 6:
        return None
    o = np.argsort(r); r, phi, z = r[o], np.unwrap(phi[o]), z[o]
    zr, pr = _fit_resid(r, z), _fit_resid(r, phi)
    a_zr, a_pr = lag1(zr), lag1(pr)
    h = len(r) // 2
    def rms(rr, zz):
        return float(np.sqrt(np.mean(_fit_resid(rr, zz) ** 2))) if len(rr) >= 3 else None
    f, s = rms(r[:h], z[:h]), rms(r[h:], z[h:])
    return a_zr, a_pr, f, s


def collect(track_iter):
    az, ap, ff, ss = [], [], [], []
    for r, phi, z in track_iter:
        m = track_measures(r, phi, z)
        if m is None:
            continue
        a_zr, a_pr, f, s = m
        if a_zr is not None: az.append(a_zr)
        if a_pr is not None: ap.append(a_pr)
        if f is not None and s is not None: ff.append(f); ss.append(s)
    ff, ss = np.array(ff), np.array(ss)
    persist = float(np.corrcoef(ff, ss)[0, 1]) if len(ff) > 10 else float("nan")
    return np.mean(az), np.mean(ap), persist, len(az)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--n_tracks", type=int, default=15000)
    ap.add_argument("--use_vertex", action="store_true")
    ap.add_argument("--max_hits", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(0)
    d = np.load(args.slice)
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    norm = {"cont_mean": d["cont_mean"], "cont_std": d["cont_std"]}
    val = rng.permutation(cont.shape[0])[:args.n_tracks]
    model = TrackerModel(norm, use_vertex=args.use_vertex).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    LM, LS = LAYER_MEANS, LAYER_STDS
    n_true = np.clip(off[val + 1] - off[val], 1, args.max_hits).astype(np.int64)
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    vtxT = torch.as_tensor(cont[val][:, [5, 6]], dtype=torch.float32, device=dev)

    real_tracks, gen_tracks = [], []
    for s in range(0, len(val), 4096):
        e = min(s + 4096, len(val))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e]); nh = torch.as_tensor(n_true[s:e], device=dev)
            vtx = vtxT[s:e] if args.use_vertex else None
            gh, _ = model.tracker.generate(ce, nh, vertex_pos=vtx)
        gh = gh.cpu().numpy()
        for j in range(e - s):
            gi = s + j; k = int(n_true[gi])
            gen_tracks.append((gh[j, :k, 0], gh[j, :k, 1], gh[j, :k, 2]))
            a, b = off[val[gi]], off[val[gi]] + k
            lc = hits[a:b, 0].astype(int).clip(0, N_LAYERS - 1)
            real_tracks.append((hits[a:b, 1] * LS[lc, 0] + LM[lc, 0],
                                hits[a:b, 2] * LS[lc, 1] + LM[lc, 1],
                                hits[a:b, 3] * LS[lc, 2] + LM[lc, 2]))
    print("=" * 66); print("TRACK RESIDUAL CORRELATION (real vs gen), tracks with n>=6"); print("=" * 66)
    print(f"{'':20s} {'lag1 z-r':>10s} {'lag1 phi-r':>11s} {'half-persist':>13s} {'ntracks':>8s}")
    for tag, tr in [("REAL", real_tracks), ("GEN", gen_tracks)]:
        azr, apr, per, n = collect(tr)
        print(f"  {tag:18s} {azr:10.3f} {apr:11.3f} {per:13.3f} {n:8d}")


if __name__ == "__main__":
    main()
