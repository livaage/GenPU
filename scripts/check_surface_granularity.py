"""Check whether a paper-style (arXiv:2512.24254) module-local coordinate representation
is viable on ColliderML: how fine is surface_id, and how tightly do hits sharing a
surface cluster in space (i.e. is the local (x,y) offset small & bounded)?

Reads ONE source tracker_hits arrow shard, bounded slice, no training.
"""
from __future__ import annotations
import glob
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

SHARD_GLOB = ("/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets/"
              "CERN___collider_ml-release-1/pileup_only_pu0_tracker_hits/0.0.0/*/*.arrow")
N_EVENTS = 4000   # bounded read (list-column: one row per event)


def main():
    shard = sorted(glob.glob(SHARD_GLOB))[0]
    print("shard:", shard.split("/")[-1])
    tbl = ipc.open_stream(pa.memory_map(shard, "r")).read_all()
    print("=== SCHEMA (per-event list columns) ===")
    for f in tbl.schema:
        print(f"  {f.name:16s} {f.type}")
    print("total events in shard:", tbl.num_rows)

    cols = [f.name for f in tbl.schema]
    want = [c for c in ("x", "y", "z", "volume_id", "layer_id", "surface_id",
                        "particle_id") if c in cols]
    sub = tbl.slice(0, min(N_EVENTS, tbl.num_rows))
    # flatten list columns across the sampled events
    d = {}
    for c in want:
        col = sub.column(c)
        d[c] = (col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
                if pa.types.is_list(col.type) else col.to_numpy(zero_copy_only=False))
    n = len(d["x"])
    print(f"\nread {n} hits from {sub.num_rows} events\n" + "=" * 70)

    x, y, z = d["x"].astype(np.float64), d["y"].astype(np.float64), d["z"].astype(np.float64)
    r = np.hypot(x, y)
    vol, lay, surf = d["volume_id"].astype(np.int64), d["layer_id"].astype(np.int64), d["surface_id"].astype(np.int64)

    # composite module keys at increasing granularity
    lay_key = vol * 1000 + lay
    surf_key = (vol.astype(np.int64) * 1_000_000 + lay * 1000 + surf)

    print("GRANULARITY (distinct ids in this slice)")
    print(f"  distinct volume_id                 : {len(np.unique(vol))}")
    print(f"  distinct (volume,layer)            : {len(np.unique(lay_key))}   <- current model's discrete anchor (~48)")
    print(f"  distinct surface_id (raw)          : {len(np.unique(surf))}")
    print(f"  distinct (volume,layer,surface)    : {len(np.unique(surf_key)):,}   <- paper's module anchor")
    print(f"  hits / module (mean)               : {n/len(np.unique(surf_key)):.1f}")

    def spread(key, name):
        """For groups of >=20 hits, physical extent within a group (std of x,y,z + radial)."""
        order = np.argsort(key, kind="stable")
        ks = key[order]; xs, ys, zs, rs = x[order], y[order], z[order], r[order]
        bnd = np.where(np.diff(ks) != 0)[0] + 1
        starts = np.concatenate([[0], bnd]); ends = np.concatenate([bnd, [len(ks)]])
        sx, sy, sz, sr, sizes = [], [], [], [], []
        for s, e in zip(starts, ends):
            if e - s < 20:
                continue
            sx.append(xs[s:e].std()); sy.append(ys[s:e].std())
            sz.append(zs[s:e].std()); sr.append(rs[s:e].std()); sizes.append(e - s)
        sx, sy, sz, sr = map(np.array, (sx, sy, sz, sr))
        print(f"\nWITHIN-{name} SPATIAL SPREAD  (std over hits sharing the id; {len(sx)} groups >=20 hits)")
        print(f"  median group size   : {int(np.median(sizes))}")
        for nm, a in (("std_x [mm]", sx), ("std_y [mm]", sy), ("std_z [mm]", sz), ("std_r [mm]", sr)):
            p = np.percentile(a, [50, 90, 99])
            print(f"  {nm:12s} median {p[0]:8.2f}   p90 {p[1]:8.2f}   p99 {p[2]:8.2f}")
        # transverse in-plane extent = the local (x,y) the paper predicts
        inplane = np.sqrt(sx**2 + sy**2)
        p = np.percentile(inplane, [50, 90, 99])
        print(f"  in-plane sqrt(sx^2+sy^2): median {p[0]:.2f}  p90 {p[1]:.2f}  p99 {p[2]:.2f} mm")

    spread(lay_key, "(volume,layer)")
    spread(surf_key, "(volume,layer,surface)")

    # radial dynamic range: how much of the drift-prone r does surface pin vs layer?
    print("\n" + "=" * 70)
    print("RADIAL SPAN comparison (what the discrete anchor already pins):")
    for key, name in ((lay_key, "layer"), (surf_key, "surface")):
        u, inv = np.unique(key, return_inverse=True)
        # mean r per group, then how spread is r WITHIN a group relative to between groups
        within = np.zeros(len(u)); cnt = np.zeros(len(u))
        for gi in range(len(u)):
            m = inv == gi
            if m.sum() >= 20:
                within[gi] = r[m].std()
        wv = within[within > 0]
        print(f"  {name:8s}: within-group r-std  median {np.median(wv):7.2f} mm   p90 {np.percentile(wv,90):7.2f} mm")


if __name__ == "__main__":
    main()
