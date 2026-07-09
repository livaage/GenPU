"""Direct measure: what fraction of tracks repeat a layer_class (real vs gen)?
Cleaner than r_mono (which used tied radii). Uses the multispecies model + norm."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel


def dup_stats(layers_list):
    has_dup, n_dup, ntot = 0, 0, 0
    for L in layers_list:
        u = len(set(L.tolist()))
        d = len(L) - u
        if d > 0:
            has_dup += 1; n_dup += d
        ntot += 1
    return has_dup / max(ntot, 1), n_dup / max(ntot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--n_tracks", type=int, default=20000)
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

    n_true = np.clip(off[val + 1] - off[val], 1, args.max_hits).astype(np.int64)
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    vtxT = torch.as_tensor(cont[val][:, [5, 6]], dtype=torch.float32, device=dev)
    real_L, gen_L = [], []
    for s in range(0, len(val), 4096):
        e = min(s + 4096, len(val))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            nh = torch.as_tensor(n_true[s:e], device=dev)
            vtx = vtxT[s:e] if args.use_vertex else None
            _, gl = model.tracker.generate(ce, nh, vertex_pos=vtx)
        gl = gl.cpu().numpy()
        for j in range(e - s):
            gi = s + j; k = int(n_true[gi])
            real_L.append(hits[off[val[gi]]:off[val[gi]] + k, 0].astype(int))
            gen_L.append(gl[j, :k].astype(int))
    for tag, L in [("REAL", real_L), ("GEN", gen_L)]:
        f, m = dup_stats(L)
        # also by whether multiplicity>1
        multi = [x for x in L if len(x) > 1]
        fm, mm = dup_stats(multi)
        print(f"{tag}: tracks={len(L)}  frac with >=1 dup layer = {f:.1%}  mean dups/track = {m:.2f}"
              f"   (among n>1 tracks: {fm:.1%})")


if __name__ == "__main__":
    main()
