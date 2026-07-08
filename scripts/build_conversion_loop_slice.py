"""Close-the-loop slice: per converting photon, store photon (for generation) +
its TRUTH e± daughters' kinematics (for a head-to-head comparison).

The loop test runs the tracker head on truth-e± vs generated-e± and compares the
resulting tracker hits — isolating CASCADE generation quality from tracker-head
error. Both go through the same head, so differences are purely the cascade.

Output npz (per converting photon; only photons with >=1 e± daughter kept):
  ph      (S, 5)  [log_E, eta, phi, vr, vz]      photon (generation conditioning)
  ee_flat (P, 6)  [log_pt, eta, phi, log_E, charge, vr(vz packed sep)]  truth e±
  ee_vz   (P,)    e± production vz
  ee_off  (S+1,)  photon s owns e± [off[s]:off[s+1]]
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pyarrow as pa


def load(sub, shard):
    base = "/scratch/gpfs/IOJALVO/lv7805/genpu_cache/CERN___collider_ml-release-1"
    f = sorted(glob.glob(f"{base}/pileup_only_pu0_{sub}/*/*/*.arrow"))[shard]
    try:
        return pa.ipc.open_file(pa.memory_map(f)).read_all()
    except pa.lib.ArrowInvalid:
        return pa.ipc.open_stream(pa.memory_map(f)).read_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/conv_loop.npz")
    args = ap.parse_args()
    ph, ee, ee_vz, lengths = [], [], [], []
    for sh in args.shards:
        t = load("particles", sh)
        pid = t.column("particle_id").to_pylist(); parent = t.column("parent_id").to_pylist()
        pdg = t.column("pdg_id").to_pylist(); energy = t.column("energy").to_pylist()
        vx = t.column("vx").to_pylist(); vy = t.column("vy").to_pylist(); vz = t.column("vz").to_pylist()
        px = t.column("px").to_pylist(); py = t.column("py").to_pylist(); pz = t.column("pz").to_pylist()
        for ev in range(t.num_rows):
            ids = np.asarray(pid[ev], np.int64); par = np.asarray(parent[ev], np.int64)
            pg = np.asarray(pdg[ev], np.int64); en = np.asarray(energy[ev], np.float64)
            rr = np.hypot(np.asarray(vx[ev]), np.asarray(vy[ev])); zz = np.asarray(vz[ev])
            pxa, pya, pza = np.asarray(px[ev]), np.asarray(py[ev]), np.asarray(pz[ev])
            pmag = np.sqrt(pxa**2 + pya**2 + pza**2) + 1e-9
            eta = np.arctanh(np.clip(pza / pmag, -1 + 1e-7, 1 - 1e-7))
            phi = np.arctan2(pya, pxa); pt = np.hypot(pxa, pya)
            idx = {int(i): k for k, i in enumerate(ids)}
            dau = {}
            for k in range(len(ids)):
                p = idx.get(int(par[k]))
                if p is not None:
                    dau.setdefault(p, []).append(k)
            for k in range(len(ids)):
                if pg[k] != 22:
                    continue
                edk = [dk for dk in dau.get(k, []) if abs(pg[dk]) == 11]
                if not edk:
                    continue
                ph.append([np.log(max(en[k], 1e-6)), float(eta[k]), float(phi[k]),
                           float(rr[k]), float(zz[k])])
                for dk in edk:
                    charge = -1.0 if pg[dk] == 11 else 1.0
                    ee.append([np.log(max(pt[dk], 1e-6)), float(eta[dk]), float(phi[dk]),
                               np.log(max(en[dk], 1e-6)), charge, float(rr[dk])])
                    ee_vz.append(float(zz[dk]))
                lengths.append(len(edk))
    ph = np.asarray(ph, np.float32); ee = np.asarray(ee, np.float32)
    ee_vz = np.asarray(ee_vz, np.float32)
    off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ph=ph, ee_flat=ee, ee_vz=ee_vz, ee_off=off)
    print(f"wrote {args.out}")
    print(f"  converting photons={len(ph)}  e± daughters={len(ee)}  (mean {len(ee)/len(ph):.2f}/photon)")


if __name__ == "__main__":
    main()
