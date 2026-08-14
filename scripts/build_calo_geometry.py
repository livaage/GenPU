"""Derive the calorimeter surfaces from the RAW source shards.

Needed because nothing downstream knows where the calo IS: `detector_geometry.py` is tracker layer
geometry, and the preprocessed stage2 calo hits carry only (eta, phi, logE, frac, detector) — no
radius. To anchor a shower's core on a helix extrapolation we must know which surface to extrapolate
to (barrel radius / endcap z, per `detector` value).

Reads `calo_hits` (x, y, z, total_energy, detector) from one raw shard and reports, per detector id:
barrel vs endcap, the radial / z position, and the depth spread. Writes a small JSON that the slice
builder and the helix anchor both read.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max_events", type=int, default=400)
    ap.add_argument("--out", default="/home/lv7805/genpu/calo_geometry.json")
    args = ap.parse_args()

    t = load_shard("calo_hits", args.shard)
    print(f"calo_hits shard {args.shard}: {t.num_rows} events, columns: {t.column_names}")

    n = min(args.max_events, t.num_rows)
    xs, ys, zs, ds, es = [], [], [], [], []
    for col, acc in [("x", xs), ("y", ys), ("z", zs), ("detector", ds)]:
        if col not in t.column_names:
            raise SystemExit(f"expected column '{col}' in calo_hits, got {t.column_names}")
    ecol = "total_energy" if "total_energy" in t.column_names else None
    for i in range(n):
        xs.append(np.asarray(t["x"][i].as_py(), np.float64))
        ys.append(np.asarray(t["y"][i].as_py(), np.float64))
        zs.append(np.asarray(t["z"][i].as_py(), np.float64))
        ds.append(np.asarray(t["detector"][i].as_py(), np.float64))
        if ecol:
            es.append(np.asarray(t[ecol][i].as_py(), np.float64))
    x, y, z, det = map(np.concatenate, (xs, ys, zs, ds))
    e = np.concatenate(es) if ecol else np.ones_like(x)
    r = np.hypot(x, y)
    print(f"{len(x)} cells from {n} events\n")

    print(f"{'det':>5} {'cells':>10} {'E frac':>7} {'r med':>9} {'r 5-95%':>15} {'|z| med':>9} "
          f"{'|z| 5-95%':>15} {'CV_r':>6} {'CV_z':>6} {'z side':>7} {'geom':>8}")
    geom = {}
    for d in np.unique(det):
        m = det == d
        rr, zz = r[m], np.abs(z[m])
        r_lo, r_hi = np.percentile(rr, [5, 95]); z_lo, z_hi = np.percentile(zz, [5, 95])
        # barrel = fixed RADIUS with depth in z; endcap = fixed |z| with extent in r. Compare
        # RELATIVE spreads: a deep forward calo has a large absolute z-spread (its own depth) yet is
        # still an endcap, which an absolute-spread test misreads as a barrel.
        cv_r = (r_hi - r_lo) / max(np.median(rr), 1e-9)
        cv_z = (z_hi - z_lo) / max(np.median(zz), 1e-9)
        barrel = cv_r < cv_z
        side = "both" if (z[m] > 0).mean() > 0.2 and (z[m] < 0).mean() > 0.2 else ("+z" if (z[m] > 0).mean() > 0.5 else "-z")
        print(f"{int(d):>5} {m.sum():>10} {e[m].sum()/e.sum():>7.3f} {np.median(rr):>9.1f} "
              f"{f'{r_lo:.0f}-{r_hi:.0f}':>15} {np.median(zz):>9.1f} {f'{z_lo:.0f}-{z_hi:.0f}':>15} "
              f"{cv_r:>6.2f} {cv_z:>6.2f} {side:>7} {'barrel' if barrel else 'endcap':>8}")
        geom[str(int(d))] = {
            "cells": int(m.sum()), "energy_frac": float(e[m].sum() / e.sum()),
            "is_barrel": bool(barrel), "cv_r": float(cv_r), "cv_z": float(cv_z), "z_side": side,
            "r_median": float(np.median(rr)), "r_p5": float(r_lo), "r_p95": float(r_hi),
            "absz_median": float(np.median(zz)), "absz_p5": float(z_lo), "absz_p95": float(z_hi),
        }

    # The anchor surface: the innermost barrel radius / smallest |z| endcap carrying real energy is
    # the calo FRONT FACE, which is what a track reaches first and what the shower core sits behind.
    bar = {k: v for k, v in geom.items() if v["is_barrel"] and v["energy_frac"] > 0.01}
    end = {k: v for k, v in geom.items() if not v["is_barrel"] and v["energy_frac"] > 0.01}
    face = {}
    if bar:
        k = min(bar, key=lambda k: bar[k]["r_p5"])
        face["barrel_r"] = bar[k]["r_p5"]; face["barrel_det"] = int(k)
    if end:
        k = min(end, key=lambda k: end[k]["absz_p5"])
        face["endcap_absz"] = end[k]["absz_p5"]; face["endcap_det"] = int(k)
    # barrel/endcap transition: |eta| where the barrel front face meets the endcap front face
    if "barrel_r" in face and "endcap_absz" in face:
        theta = np.arctan2(face["barrel_r"], face["endcap_absz"])
        face["eta_transition"] = float(-np.log(np.tan(theta / 2)))
    print(f"\nanchor surface (calo front face): {json.dumps(face, indent=2)}")

    out = {"shard": args.shard, "events": n, "detectors": geom, "front_face": face}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
