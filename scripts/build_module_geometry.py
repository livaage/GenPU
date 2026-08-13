"""Build the tracker MODULE vocabulary + per-module local frame — the surface-granular
analogue of detector_geometry.LAYER_GEOMETRY, for the paper-style (arXiv:2512.24254)
surface-local representation.

A "module" is the (volume_id, layer_id, surface_id) composite (surface_id is reused across
layers, so the composite is the real module key). For each module we store, from data:
  - mean (x, y, z, time)  -> the local-frame ORIGIN
  - std  (x, y, z, time)  -> per-axis standardization (so local residuals are ~N(0,1))
  - count, and the parent layer_class (0..47) for optional hierarchical modelling.

Pre-build tests (scripts/tracker_surface_prebuild.py) showed within-module spread is bounded
(<=~50 mm all volumes), so raw mean-subtracted Cartesian offsets are a valid small local frame.

Reads source arrow shards directly (CPU only). Output: module_geometry.npz.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, explode_list_columns, build_event_index
from genpu.detector_geometry import LAYER_TO_CLASS

# collision-free int64 module key: vol(uint8)<<48 | layer(uint16)<<32 | surface(uint32).
# MUST stay identical to genpu.module_geometry.module_composite.
def module_composite(vol, layer, surface):
    return (vol.astype(np.int64) << 48) | (layer.astype(np.int64) << 32) | surface.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/module_geometry.npz")
    ap.add_argument("--std_floor", type=float, default=0.05, help="min per-axis std [mm/ns] to avoid /0")
    ap.add_argument("--max_events", type=int, default=0, help="0 = all events in each shard")
    args = ap.parse_args()

    # global accumulators keyed by int64 composite -> [count, sx,sy,sz,st, sxx,syy,szz,stt]
    acc: dict[int, np.ndarray] = {}
    lay_of: dict[int, int] = {}

    for sh in args.shards:
        t = load_shard("tracker_hits", sh)
        idx = build_event_index(t)
        eids = sorted(idx.keys())
        if args.max_events:
            eids = eids[:args.max_events]
        n_hits = 0
        for eid in eids:
            ev = explode_list_columns(t, idx[eid])
            x, y, z = ev["x"].astype(np.float64), ev["y"].astype(np.float64), ev["z"].astype(np.float64)
            tm = ev["time"].astype(np.float64) if "time" in ev else np.zeros_like(x)
            vol, lay, surf = ev["volume_id"], ev["layer_id"], ev["surface_id"]
            comp = module_composite(np.asarray(vol), np.asarray(lay), np.asarray(surf))
            u, inv = np.unique(comp, return_inverse=True)
            cnt = np.bincount(inv).astype(np.float64)
            sx = np.bincount(inv, weights=x); sy = np.bincount(inv, weights=y)
            sz = np.bincount(inv, weights=z); st = np.bincount(inv, weights=tm)
            sxx = np.bincount(inv, weights=x * x); syy = np.bincount(inv, weights=y * y)
            szz = np.bincount(inv, weights=z * z); stt = np.bincount(inv, weights=tm * tm)
            vol_i, lay_i = np.asarray(vol).astype(int), np.asarray(lay).astype(int)
            for k, key in enumerate(u):
                key = int(key)
                blk = np.array([cnt[k], sx[k], sy[k], sz[k], st[k], sxx[k], syy[k], szz[k], stt[k]])
                if key in acc:
                    acc[key] += blk
                else:
                    acc[key] = blk
                    # record parent layer_class from the first hit of this module
                    first = np.nonzero(inv == k)[0][0]
                    lay_of[key] = LAYER_TO_CLASS.get((vol_i[first], lay_i[first]), 0)
            n_hits += len(x)
        print(f"  shard {sh}: {len(eids)} events, {n_hits:,} hits, running modules={len(acc):,}", flush=True)

    keys = np.array(sorted(acc.keys()), dtype=np.int64)
    M = len(keys)
    mean = np.zeros((M, 4), np.float32); std = np.zeros((M, 4), np.float32)
    count = np.zeros(M, np.int64); lay_cls = np.zeros(M, np.int32)
    for i, key in enumerate(keys):
        c, sx, sy, sz, st, sxx, syy, szz, stt = acc[int(key)]
        m = np.array([sx, sy, sz, st]) / c
        v = np.array([sxx, syy, szz, stt]) / c - m ** 2
        mean[i] = m
        std[i] = np.sqrt(np.clip(v, 0, None))
        count[i] = int(c)
        lay_cls[i] = lay_of[int(key)]
    std = np.maximum(std, args.std_floor)

    # decode key -> (vol, layer, surface) for reconstruction / inspection
    vol = (keys >> 48) & 0xFF
    layer = (keys >> 32) & 0xFFFF
    surface = keys & 0xFFFFFFFF
    mod_key = np.stack([vol, layer, surface], axis=1).astype(np.int64)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, module_composite=keys, module_key=mod_key,
                        module_mean=mean, module_std=std, module_count=count,
                        module_layer_class=lay_cls)
    print(f"\nwrote {out}")
    print(f"  modules: {M:,}  (from shards {args.shards})")
    print(f"  hits total: {count.sum():,}   hits/module median={int(np.median(count))} min={count.min()}")
    print(f"  local std [mm] x: med {np.median(std[:,0]):.2f} p99 {np.percentile(std[:,0],99):.2f}  "
          f"y: med {np.median(std[:,1]):.2f}  z: med {np.median(std[:,2]):.2f}")
    print(f"  modules with <20 hits: {(count<20).sum()} ({(count<20).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
