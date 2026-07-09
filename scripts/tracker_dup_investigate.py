"""Investigate WHERE/WHY generated tracks tie in r (the r_mono tail).
Distinguishes physical same-r hits (same layer, different phi/z — legit, e.g. barrel)
from FULLY-IDENTICAL hits (same layer,r,phi,z — a real defect). By barrel/endcap.
Uses the multispecies model + its own norm."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS, LAYER_IS_BARREL


def analyse(layers, r, phi, z):
    """Per-track: counts of full-identical pairs, same-r-diff-posn pairs, by barrel/endcap."""
    full_id = same_r = 0
    fi_barrel = fi_endcap = 0
    lay = layers.astype(int)
    for L in np.unique(lay):
        idx = np.where(lay == L)[0]
        if len(idx) < 2:
            continue
        rr, pp, zz = r[idx], phi[idx], z[idx]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                dr = abs(rr[i] - rr[j]); dpos = abs(pp[i] - pp[j]) + abs(zz[i] - zz[j])
                if dr < 0.5 and dpos < 1.0:       # essentially identical hit
                    full_id += 1
                    if LAYER_IS_BARREL[min(L, N_LAYERS - 1)]:
                        fi_barrel += 1
                    else:
                        fi_endcap += 1
                elif dr < 0.5:                     # same r, different phi/z (physical)
                    same_r += 1
    return full_id, same_r, fi_barrel, fi_endcap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--n_tracks", type=int, default=12000)
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
    tot = {"REAL": [0, 0, 0, 0, 0], "GEN": [0, 0, 0, 0, 0]}  # full_id, same_r, fi_bar, fi_end, ntracks
    for s in range(0, len(val), 4096):
        e = min(s + 4096, len(val))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e]); nh = torch.as_tensor(n_true[s:e], device=dev)
            vtx = vtxT[s:e] if args.use_vertex else None
            gh, gl = model.tracker.generate(ce, nh, vertex_pos=vtx)
        gh = gh.cpu().numpy(); gl = gl.cpu().numpy()
        for j in range(e - s):
            gi = s + j; k = int(n_true[gi])
            fi, sr, fb, fe = analyse(gl[j, :k], gh[j, :k, 0], gh[j, :k, 1], gh[j, :k, 2])
            for t, v in zip(tot["GEN"], [fi, sr, fb, fe, 1]): pass
            tot["GEN"] = [a + b for a, b in zip(tot["GEN"], [fi, sr, fb, fe, 1])]
            a, b = off[val[gi]], off[val[gi]] + k
            lc = hits[a:b, 0].astype(int).clip(0, N_LAYERS - 1)
            rr = hits[a:b, 1] * LS[lc, 0] + LM[lc, 0]; pp = hits[a:b, 2] * LS[lc, 1] + LM[lc, 1]
            zz = hits[a:b, 3] * LS[lc, 2] + LM[lc, 2]
            fi, sr, fb, fe = analyse(lc, rr, pp, zz)
            tot["REAL"] = [a2 + b2 for a2, b2 in zip(tot["REAL"], [fi, sr, fb, fe, 1])]
    print("=" * 60); print("TRACKER DUPLICATE INVESTIGATION"); print("=" * 60)
    for tag in ("REAL", "GEN"):
        fi, sr, fb, fe, nt = tot[tag]
        print(f"{tag}: tracks={nt}  full-identical pairs/track={fi/nt:.3f} (barrel {fb/nt:.3f}, endcap {fe/nt:.3f})"
              f"   same-r-diff-posn pairs/track={sr/nt:.3f}")


if __name__ == "__main__":
    main()
