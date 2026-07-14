"""Lightweight per-particle COUNT slice: conditioning + geometric d0 + n_hits, all species.
d0 = vx*sin(phi_p) - vy*cos(phi_p)  (production transverse impact parameter; phi-invariant,
defined for neutrals too, computable at generation from any vertex+momentum). No hits stored."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, explode_list_columns
from genpu.preprocessing import pdg_to_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/count.npz")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    cont_l, d0_l, nh_l, pdg_l = [], [], [], []
    for sh in args.shards:
        P = load_shard("particles", sh); T = load_shard("tracker_hits", sh)
        for e in range(min(P.num_rows, T.num_rows)):
            pe = explode_list_columns(P, e); te = explode_list_columns(T, e)
            tpid = te["particle_id"].astype(np.int64)
            pt = np.hypot(pe["px"], pe["py"]); p = np.sqrt(pt**2 + pe["pz"]**2)
            eta = np.arcsinh(pe["pz"] / (pt + 1e-9)); phip = np.arctan2(pe["py"], pe["px"])
            log_pt = np.log(pt + 1e-6); log_E = np.log(np.clip(pe["energy"], 1e-6, None))
            vr = np.hypot(pe["vx"], pe["vy"])
            d0 = pe["vx"] * np.sin(phip) - pe["vy"] * np.cos(phip)
            cls = pdg_to_class(pe["pdg_id"].astype(np.int64))
            for i in range(len(pe["particle_id"])):
                nh = int((tpid == pe["particle_id"][i]).sum())
                if nh == 0:
                    continue
                cont_l.append([log_pt[i], eta[i], log_E[i], pe["charge"][i], pe["mass"][i], vr[i], pe["vz"][i]])
                d0_l.append(d0[i]); nh_l.append(nh); pdg_l.append(float(cls[i]))
        print(f"  shard {sh}: total {len(cont_l)}", flush=True)
    cont = np.asarray(cont_l, np.float32); d0 = np.asarray(d0_l, np.float32)
    nh = np.asarray(nh_l, np.int64); pdg = np.asarray(pdg_l, np.float32)
    norm = {"cont_mean": cont.mean(0), "cont_std": cont.std(0) + 1e-6,
            "d0_mean": np.array([np.abs(d0).mean()], np.float32), "d0_std": np.array([np.abs(d0).std() + 1e-6], np.float32)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, d0=d0, n_hits=nh, pdg=pdg, **norm)
    print(f"wrote {out}  particles={len(nh)}  n_hits mean={nh.mean():.2f} median={np.median(nh):.0f}  "
          f"|d0| median={np.median(np.abs(d0)):.2f}")


if __name__ == "__main__":
    main()
