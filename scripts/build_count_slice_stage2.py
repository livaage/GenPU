"""Count slice from the CORRECT preprocessed stage2 (event-joined): per-particle cont + d0 +
n_hits + pdg + primary flag. d0 = vx*sin(phi) - vy*cos(phi) (geometric impact parameter)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/count_stage2.npz")
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    cont_l, d0_l, nh_l, pdg_l, prim_l = [], [], [], [], []
    for sh in args.shards:
        p = Path(args.preproc_dir) / f"shard_{sh:04d}_stage2.npz"
        if not p.exists():
            print(f"  missing {p}"); continue
        d = np.load(p); pf, aux, off = d["particle_features"], d["particle_aux"], d["tracker_offsets"]
        nh = np.diff(off); keep = nh >= 1
        vr = np.hypot(aux[:, AUX_VX], aux[:, AUX_VY]); logE = np.log(np.clip(aux[:, AUX_ENERGY], 1e-6, None))
        d0 = aux[:, AUX_VX] * np.sin(pf[:, PF_PHI]) - aux[:, AUX_VY] * np.cos(pf[:, PF_PHI])
        cont = np.stack([pf[:, PF_LOGPT], pf[:, PF_ETA], logE, pf[:, PF_CHARGE], pf[:, PF_MASS], vr, aux[:, AUX_VZ]], 1)
        cont_l.append(cont[keep].astype(np.float32)); d0_l.append(d0[keep].astype(np.float32))
        nh_l.append(nh[keep].astype(np.int64)); pdg_l.append(pf[keep, PF_PDG].astype(np.float32))
        prim_l.append((aux[keep, AUX_PRIMARY] > 0.5).astype(np.float32))
        print(f"  shard {sh}: +{keep.sum()}", flush=True)
    cont = np.concatenate(cont_l); d0 = np.concatenate(d0_l); nh = np.concatenate(nh_l)
    pdg = np.concatenate(pdg_l); prim = np.concatenate(prim_l)
    norm = {"cont_mean": cont.mean(0), "cont_std": cont.std(0) + 1e-6,
            "d0_mean": np.array([np.abs(d0).mean()], np.float32), "d0_std": np.array([np.abs(d0).std() + 1e-6], np.float32)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, d0=d0, n_hits=nh, pdg=pdg, primary=prim, **norm)
    print(f"wrote {out}  n={len(nh)}  PIONS n_hits median={np.median(nh[np.isin(pdg,[3,4])]):.0f}  |d0| median={np.median(np.abs(d0)):.2f}")


if __name__ == "__main__":
    main()
