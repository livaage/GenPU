"""What IS the calo cell grid? Needed before a cell-level metric can exist.

Real calo hits sit at discrete cell centres; generated showers are continuous (eta, phi) points.
A partition-invariant metric has to put BOTH on the same cells, so we need the grid: how many
distinct cells, whether (eta, phi) is separable per detector, and the spacing.

Reports per detector so the metric can snap generated points with searchsorted rather than a
nearest-neighbour search over millions of cells.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, build_event_index, explode_list_columns  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n_events", type=int, default=200)
ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/cell_grid.json")
a = ap.parse_args()

ca = load_shard("calo_hits", a.shard)
ci = build_event_index(ca)
X, Y, Z, D = [], [], [], []
for eid in sorted(ci)[:a.n_events]:
    c = explode_list_columns(ca, ci[eid])
    X.append(np.asarray(c["x"], np.float64)); Y.append(np.asarray(c["y"], np.float64))
    Z.append(np.asarray(c["z"], np.float64)); D.append(np.asarray(c["detector"], np.int64))
x, y, z, d = map(np.concatenate, (X, Y, Z, D))
r = np.hypot(x, y); theta = np.arctan2(r, z)
eta = -np.log(np.tan(np.clip(theta, 1e-9, np.pi - 1e-9) / 2)); phi = np.arctan2(y, x)
print(f"cells sampled {len(x):,} over {a.n_events} events\n")
out = {}
print(f"{'det':>4} {'cells':>10} {'uniq eta':>9} {'uniq phi':>9} {'d_eta med':>10} {'d_phi med':>10} "
      f"{'separable?':>11}")
for det in np.unique(d):
    m = d == det
    ue = np.unique(np.round(eta[m], 5)); up = np.unique(np.round(phi[m], 5))
    # separable = the observed (eta,phi) pairs are a small fraction of the full outer product,
    # i.e. a real grid rather than an irregular tiling
    pairs = len(np.unique(np.round(np.stack([eta[m], phi[m]], 1), 5), axis=0))
    sep = pairs / max(len(ue) * len(up), 1)
    de = np.median(np.diff(ue)) if len(ue) > 1 else np.nan
    dp = np.median(np.diff(up)) if len(up) > 1 else np.nan
    out[int(det)] = dict(cells=int(m.sum()), uniq_eta=len(ue), uniq_phi=len(up), pairs=int(pairs),
                         d_eta=float(de), d_phi=float(dp), fill=float(sep))
    print(f"{det:>4} {m.sum():>10,} {len(ue):>9,} {len(up):>9,} {de:>10.5f} {dp:>10.5f} "
          f"{sep:>10.3f}{'*' if sep > 0.5 else ''}")
print("\n* fill ~1 => (eta,phi) is a clean separable grid; small fill => irregular tiling, "
      "snap must be per-pair not per-axis")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"wrote {a.out}")
