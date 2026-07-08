"""Cascade spike: build a per-photon conversion slice from pu0 truth.

For each photon: conditioning [log_E, eta, vr, vz] and targets
  converted   (bool)   — has >=1 e± daughter (recorded conversion)
  log_conv_r  (float)  — log median e± production radius, if converted
  esplit      (float)  — leading-e± energy fraction (0.5..1), if >=2 e daughters
This is the target the conversion GENERATOR must reproduce (converts?/where/split),
conditioned only on the photon (noise-free input, per the energy-head lesson).
"""
from __future__ import annotations
import argparse, glob
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
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/photon_conv.npz")
    args = ap.parse_args()

    cond, converted, log_conv_r, esplit = [], [], [], []
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
                if pg[k] != 22:
                    continue
                cond.append([np.log(max(en[k], 1e-6)), float(eta[k]), float(r[k]), float(zz[k])])
                ee = [dk for dk in dau.get(k, []) if abs(pg[dk]) == 11]
                if len(ee) >= 1:
                    converted.append(1.0)
                    log_conv_r.append(float(np.log(max(np.median([r[dk] for dk in ee]), 1.0))))
                    if len(ee) >= 2:
                        es = sorted([en[dk] for dk in ee], reverse=True)
                        esplit.append(float(es[0] / max(sum(es), 1e-9)))
                    else:
                        esplit.append(np.nan)
                else:
                    converted.append(0.0); log_conv_r.append(np.nan); esplit.append(np.nan)

    cond = np.asarray(cond, np.float32)
    converted = np.asarray(converted, np.float32)
    log_conv_r = np.asarray(log_conv_r, np.float32)
    esplit = np.asarray(esplit, np.float32)
    cm, cs = cond.mean(0), cond.std(0) + 1e-6
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, cond=cond, converted=converted, log_conv_r=log_conv_r,
                        esplit=esplit, cond_mean=cm, cond_std=cs)
    conv = converted > 0.5
    print(f"wrote {args.out}")
    print(f"  photons={len(cond)}  conv_frac={conv.mean():.1%}")
    print(f"  log_conv_r (converted): mean={np.nanmean(log_conv_r):.2f} std={np.nanstd(log_conv_r):.2f} "
          f"(r median={np.exp(np.nanmedian(log_conv_r)):.0f}mm)")
    print(f"  esplit: mean={np.nanmean(esplit):.3f} std={np.nanstd(esplit):.3f}  (n={np.sum(~np.isnan(esplit))})")


if __name__ == "__main__":
    main()
