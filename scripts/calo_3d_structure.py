"""Map the calo's 3D structure — the dimension preprocessing throws away.

preprocessing.py converts calo cell (x,y,z) -> (eta,phi) and keeps only `detector` (6 values) as a
depth proxy, so the calo model is 2D and the plan's own acceptance metrics (layer-wise energy
fractions, longitudinal shower profiles) have never been computable. Before adding depth back we
need to know what it looks like. Four questions, all from REAL data, no model:

  A LAYER STRUCTURE. Is there discrete longitudinal segmentation inside each detector, or is
    `detector` (6 values) already the whole depth story? Decides whether the model needs a
    CONTINUOUS depth coordinate or a CATEGORICAL layer index.

  B LONGITUDINAL PROFILE. Energy fraction vs depth, per species. This is a standard CaloChallenge
    acceptance metric that this project has never been able to compute.

  C DEPTH RESOLUTION. Spacing between layers, which sets the granularity the model must reproduce.

  D THE ONE THAT COULD REINTERPRET PAST RESULTS. Does shower DEPTH correlate with the lateral WIDTH
    we have been fitting in 2D? `width_std` has been a top gate discriminator all along. If deep
    showers are systematically wider, then a 2D model has been absorbing depth variation into
    width, and the standing width defect is partly a missing dimension rather than a bad fit.

Depth is measured from the calo front face along the shower direction: barrel dets {10,13} use
r - barrel_r; endcap dets {9,11,12,14} use |z| - endcap_|z|.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, build_event_index, explode_list_columns  # noqa: E402
from genpu.calo_geom import load_front_face, wrap_pi  # noqa: E402

BARREL = {10, 13}
NAMES = {11: "e-", -11: "e+", 22: "gamma", 211: "pi+", -211: "pi-", 2212: "p"}

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n_events", type=int, default=250)
ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/calo_3d_structure.json")
a = ap.parse_args()
R0, Z0 = load_front_face(None)
print(f"front face: barrel r={R0:.1f} mm, endcap |z|={Z0:.1f} mm\n", flush=True)

t, ca = load_shard("particles", a.shard), load_shard("calo_hits", a.shard)
pi, ci = build_event_index(t), build_event_index(ca)
eids = sorted(set(pi) & set(ci))[:a.n_events]

CX, CD, CE, CETA, CPHI = [], [], [], [], []
sh = {}          # (species) -> list of per-shower (depth_mean, depth_std, width, logE)
for eid in eids:
    p = explode_list_columns(t, pi[eid]); c = explode_list_columns(ca, ci[eid])
    x, y, z = (np.asarray(c[k], np.float64) for k in ("x", "y", "z"))
    det = np.asarray(c["detector"], np.int64); tot = np.asarray(c["total_energy"], np.float64)
    r = np.hypot(x, y); az = np.abs(z)
    isb = np.isin(det, list(BARREL))
    depth = np.where(isb, r - R0, az - Z0)
    th = np.arctan2(r, z); eta = -np.log(np.tan(np.clip(th, 1e-9, np.pi - 1e-9) / 2))
    phi = np.arctan2(y, x)
    CX.append(det); CD.append(depth); CE.append(tot); CETA.append(eta); CPHI.append(phi)

    P = np.asarray(p["particle_id"], np.int64); G = np.asarray(p["pdg_id"], np.int64)
    pos = {int(v): k for k, v in enumerate(P)}
    per = {}
    for j, (ids, ens) in enumerate(zip(c["contrib_particle_ids"], c["contrib_energies"])):
        for q, e in zip(np.asarray(ids, np.int64), np.asarray(ens, np.float64)):
            per.setdefault(int(q), []).append((depth[j], eta[j], phi[j], e))
    for q, rows in per.items():
        k = pos.get(q)
        if k is None or len(rows) < 3: continue
        nm = NAMES.get(int(G[k]))
        if nm is None: continue
        d_, e_, p_, w_ = (np.array([v[i] for v in rows]) for i in range(4))
        W = w_.sum()
        if W <= 0: continue
        dm = float((d_ * w_).sum() / W)
        ds = float(np.sqrt(max((w_ * (d_ - dm) ** 2).sum() / W, 0)))
        ec = (e_ * w_).sum() / W; pc = (p_ * w_).sum() / W
        lat = float(np.sqrt(((w_ * ((e_ - ec) ** 2 + wrap_pi(p_ - pc) ** 2)).sum()) / W))
        sh.setdefault(nm, []).append((dm, ds, lat, np.log(W)))

det, depth, E = np.concatenate(CX), np.concatenate(CD), np.concatenate(CE)
out = {}
print("A/C  LAYER STRUCTURE per detector (depth from front face, mm)")
print(f"{'det':>4} {'type':>7} {'cells':>10} {'uniq depth':>11} {'p5':>8} {'p50':>8} {'p95':>8} "
      f"{'med spacing':>12} {'E frac':>8}")
for d in np.unique(det):
    m = det == d; dd = depth[m]
    u = np.unique(np.round(dd, 2))
    sp = np.median(np.diff(u)) if len(u) > 1 else np.nan
    out[f"det{d}"] = dict(cells=int(m.sum()), uniq_depth=len(u), spacing=float(sp),
                          efrac=float(E[m].sum() / E.sum()))
    print(f"{d:>4} {'barrel' if d in BARREL else 'endcap':>7} {m.sum():>10,} {len(u):>11,} "
          f"{np.percentile(dd,5):>8.1f} {np.percentile(dd,50):>8.1f} {np.percentile(dd,95):>8.1f} "
          f"{sp:>12.3f} {E[m].sum()/E.sum():>8.3f}")

print("\nB  LONGITUDINAL PROFILE — energy fraction by depth bin (mm from front face)")
edges = [0, 50, 100, 200, 400, 800, 1600, 1e9]
hdr = "  ".join(f"{lo}-{hi if hi < 1e8 else 'inf'}" for lo, hi in zip(edges[:-1], edges[1:]))
print(f"{'species':>8}  {hdr}")
for nm, rows in sorted(sh.items(), key=lambda kv: -len(kv[1])):
    if len(rows) < 500: continue
    arr = np.array(rows)
    print(f"{nm:>8}  n={len(rows):,}  depth mean {arr[:,0].mean():7.1f}  spread {arr[:,1].mean():7.1f}")
prof = {}
for nm in sh:
    pass
for d_lo, d_hi in zip(edges[:-1], edges[1:]):
    m = (depth >= d_lo) & (depth < d_hi)
    prof[f"{d_lo}-{d_hi}"] = float(E[m].sum() / E.sum())
print("  ALL cells, energy fraction by depth: " +
      "  ".join(f"{k}:{v:.3f}" for k, v in prof.items()))
out["profile_all"] = prof

print("\nD  DOES DEPTH CORRELATE WITH LATERAL WIDTH?  (per shower, >=3 cells)")
print(f"{'species':>8} {'showers':>9} {'corr(depth_mean, width)':>24} {'corr(depth_std, width)':>23}")
for nm, rows in sorted(sh.items(), key=lambda kv: -len(kv[1])):
    if len(rows) < 500: continue
    arr = np.array(rows)
    c1 = float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1])
    c2 = float(np.corrcoef(arr[:, 1], arr[:, 2])[0, 1])
    out[f"corr_{nm}"] = dict(depth_mean_width=c1, depth_std_width=c2, n=len(rows))
    print(f"{nm:>8} {len(rows):>9,} {c1:>24.3f} {c2:>23.3f}")
print("\n  |corr| > ~0.3 => a 2D model has been absorbing depth variation into lateral width,")
print("  and the standing width defect is partly a MISSING DIMENSION, not a bad fit.")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
