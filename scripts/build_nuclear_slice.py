"""Cascade hard-case spike: per-hadron nuclear-interaction slice from pu0 truth.

For each charged pion (default): conditioning [log_E, eta, vr, vz] and targets
  interacted  (bool)  — has >=1 displaced daughter (r_dau > r_parent+20mm)
  log_int_r   (float) — log(min displaced-daughter radius), if interacted
  mult        (int)   — # displaced daughters (the variable-multiplicity target)
  efrac       (float) — sum(E_dau)/E_parent, if interacted
The variable multiplicity is the new challenge vs photon conversions (always 2).
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
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--parents", type=int, nargs="+", default=[211, -211])
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/pion_nuclear.npz")
    args = ap.parse_args()
    absset = set(abs(p) for p in args.parents)

    cond, inter, log_int_r, mult, efrac = [], [], [], [], []
    for sh in args.shards:
        t = load("particles", sh)
        pid = t.column("particle_id").to_pylist(); parent = t.column("parent_id").to_pylist()
        pdg = t.column("pdg_id").to_pylist(); energy = t.column("energy").to_pylist()
        vx = t.column("vx").to_pylist(); vy = t.column("vy").to_pylist(); vz = t.column("vz").to_pylist()
        px = t.column("px").to_pylist(); py = t.column("py").to_pylist(); pz = t.column("pz").to_pylist()
        for ev in range(t.num_rows):
            ids = np.asarray(pid[ev], np.int64); par = np.asarray(parent[ev], np.int64)
            pg = np.asarray(pdg[ev], np.int64); en = np.asarray(energy[ev], np.float64)
            r = np.hypot(np.asarray(vx[ev]), np.asarray(vy[ev])); zz = np.asarray(vz[ev])
            pxa, pya, pza = np.asarray(px[ev]), np.asarray(py[ev]), np.asarray(pz[ev])
            pmag = np.sqrt(pxa**2 + pya**2 + pza**2) + 1e-9
            eta = np.arctanh(np.clip(pza / pmag, -1 + 1e-7, 1 - 1e-7))
            idx = {int(i): k for k, i in enumerate(ids)}
            dau = {}
            for k in range(len(ids)):
                p = idx.get(int(par[k]))
                if p is not None:
                    dau.setdefault(p, []).append(k)
            for k in range(len(ids)):
                if abs(pg[k]) not in absset:
                    continue
                cond.append([np.log(max(en[k], 1e-6)), float(eta[k]), float(r[k]), float(zz[k])])
                disp = [dk for dk in dau.get(k, []) if r[dk] > r[k] + 20]
                if disp:
                    inter.append(1.0)
                    log_int_r.append(float(np.log(max(np.min([r[dk] for dk in disp]), 1.0))))
                    mult.append(len(disp))
                    efrac.append(float(sum(en[dk] for dk in disp) / max(en[k], 1e-9)))
                else:
                    inter.append(0.0); log_int_r.append(np.nan); mult.append(0); efrac.append(np.nan)

    cond = np.asarray(cond, np.float32); inter = np.asarray(inter, np.float32)
    log_int_r = np.asarray(log_int_r, np.float32); mult = np.asarray(mult, np.int64)
    efrac = np.asarray(efrac, np.float32)
    cm, cs = cond.mean(0), cond.std(0) + 1e-6
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, cond=cond, inter=inter, log_int_r=log_int_r,
                        mult=mult, efrac=efrac, cond_mean=cm, cond_std=cs)
    m = inter > 0.5
    print(f"wrote {args.out}")
    print(f"  hadrons={len(cond)}  interact_frac={m.mean():.1%}")
    print(f"  mult (interacted): mean={mult[m].mean():.2f} median={np.median(mult[m]):.0f} max={mult[m].max()}")
    print(f"  log_int_r mean={np.nanmean(log_int_r):.2f} (median r={np.exp(np.nanmedian(log_int_r)):.0f}mm)")
    print(f"  efrac median={np.nanmedian(efrac):.2f}")


if __name__ == "__main__":
    main()
