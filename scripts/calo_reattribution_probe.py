"""J3a — what WOULD re-attributing calo cells to the calo-incident ancestor change?

Measured 2026-08-24: 64.8% of calo-depositing particles were born INSIDE the calorimeter (past the
front face) and carry 54.2% of all cells. `build_calo_slice.py` groups by `calo_offsets`, i.e. per
DIRECT depositor, so those shower FRAGMENTS are trained as if each were its own shower with its own
"incident particle" kinematics. Species ordering tracks calo gate difficulty: photon 0.1% split
(best gate 0.557), e± 74% (worst).

This probe answers, with NO training, whether the fix does what the hypothesis predicts:
  (1) SHOWER STATISTICS — merging fragments into their calo-incident ancestor should sharply cut
      the e± shower count and raise cells/shower, and barely touch photons (the control).
  (2) THE ANCHOR — a second, possibly larger defect. The Phase 1 helix anchor extrapolates a
      particle's trajectory TO the calo front face. For a particle BORN INSIDE the calo that is
      meaningless (it is already past the face), so we report the anchor-branch mix separately for
      born-inside vs born-outside depositors. If born-inside showers are getting turning-point or
      degenerate anchors, the anchored frame they were trained in was never physical.

Needs the complete parent graph (shard_XXXX_graph.npz) — the chain runs through INVISIBLE particles
(16.1% of direct edges), so stage2 alone cannot do this walk.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.calo_geom import core_anchor, load_front_face  # noqa: E402

NAMES = {11: "e-", -11: "e+", 22: "gamma", 211: "pi+", -211: "pi-", 2212: "p", 2112: "n"}
MODE = {0: "barrel", 1: "endcap", 2: "turning", 3: "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--graph_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade_graph")
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/reattribution_probe.json")
    a = ap.parse_args()

    R, Z = load_front_face(a.geometry)
    print(f"front face: barrel r={R:.1f} mm, endcap |z|={Z:.1f} mm", flush=True)
    g = np.load(f"{a.graph_dir}/shard_{a.shard:04d}_graph.npz")
    ev, pid, par = g["event_id"], g["particle_id"], g["parent_id"]
    pdg, ncal = g["pdg_id"], g["n_calo_hits"]
    vx, vy, vz = g["vx"], g["vy"], g["vz"]
    px, py, pz = g["px"], g["py"], g["pz"]
    chg = g["charge"]
    n = len(pid)
    vr = np.hypot(vx, vy)
    inside = (vr >= R) | (np.abs(vz) >= Z)          # born past the calo front face
    dep = ncal >= 1
    print(f"nodes {n:,}   depositors {dep.sum():,}   born-inside among depositors "
          f"{inside[dep].mean():.3f}   their cell share {ncal[dep & inside].sum()/ncal[dep].sum():.3f}",
          flush=True)

    # ---- walk each depositor up to its calo-INCIDENT ancestor (first ancestor born OUTSIDE) ----
    key = (ev.astype(np.int64) << 32) | pid.astype(np.int64)
    order = np.argsort(key); ks = key[order]
    def row_of(e, p):
        k = (e.astype(np.int64) << 32) | p.astype(np.int64)
        i = np.clip(np.searchsorted(ks, k), 0, len(ks) - 1)
        return np.where(ks[i] == k, order[i], -1)

    idx = np.where(dep)[0]
    cur = idx.copy()
    anc = idx.copy()
    live = inside[cur]                      # only born-inside need lifting
    for step in range(32):
        if not live.any():
            break
        nxt = row_of(ev[cur[live]], par[cur[live]])
        good = nxt >= 0
        upd = np.where(live)[0][good]
        cur[upd] = nxt[good]
        anc[upd] = nxt[good]
        stalled = np.where(live)[0][~good]
        live[stalled] = False               # parent missing: keep as-is
        live[upd] = inside[cur[upd]]        # keep lifting while still born-inside
    print(f"  lifted {(anc != idx).sum():,} of {len(idx):,} depositors "
          f"({(anc != idx).mean():.3f}); still born-inside after walk: {inside[anc].mean():.3f}",
          flush=True)

    # ---- (1) shower statistics, before vs after ----
    res = {}
    print(f"\n{'species':>8} {'showers BEFORE':>15} {'showers AFTER':>14} {'ratio':>7} "
          f"{'cells/sh BEFORE':>16} {'cells/sh AFTER':>15}")
    for code, nm in NAMES.items():
        m = idx[pdg[idx] == code]
        if len(m) < 2000:
            continue
        am = anc[pdg[idx] == code]
        uniq, inv = np.unique((ev[am].astype(np.int64) << 32) | pid[am].astype(np.int64),
                              return_inverse=True)
        merged = np.bincount(inv, weights=ncal[m].astype(np.float64))
        before_n, after_n = len(m), len(uniq)
        res[nm] = dict(showers_before=int(before_n), showers_after=int(after_n),
                       cells_per_shower_before=float(ncal[m].mean()),
                       cells_per_shower_after=float(merged.mean()),
                       born_inside_frac=float(inside[m].mean()))
        print(f"{nm:>8} {before_n:>15,} {after_n:>14,} {after_n/before_n:>7.3f} "
              f"{ncal[m].mean():>16.2f} {merged.mean():>15.2f}")

    # ---- (2) anchor branch mix: born-inside vs born-outside depositors ----
    print(f"\nANCHOR BRANCH of the CURRENT per-depositor frame (helix extrapolated to the face):")
    print(f"{'population':>22} {'n':>10} " + " ".join(f"{MODE[k]:>9}" for k in range(4)))
    pt = np.hypot(px, py)
    phi0 = np.arctan2(py, px)
    eta = np.arcsinh(np.clip(pz / np.clip(pt, 1e-9, None), -30, 30))
    for lab, sel in [("depositors born OUTSIDE", idx[~inside[idx]]),
                     ("depositors born INSIDE", idx[inside[idx]])]:
        s = sel if len(sel) < 400000 else np.random.default_rng(0).choice(sel, 400000, replace=False)
        _, _, mode = core_anchor(pt[s], phi0[s], eta[s], chg[s], vx[s], vy[s], vz[s], R, Z)
        frac = [float(np.mean(mode == k)) for k in range(4)]
        res[f"anchor_{lab.split()[-1].lower()}"] = frac
        print(f"{lab:>22} {len(s):>10,} " + " ".join(f"{f:>9.3f}" for f in frac))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
