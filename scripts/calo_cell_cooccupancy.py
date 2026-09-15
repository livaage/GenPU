"""WHERE does the generator put two deposits in one readout channel?

Job 13929818 projected generated points onto real ODD cells and found a collision rate of 0.0361 --
3.6% of generated points land close enough together that the detector would report one cell. The
real floor through the identical path is 0.0003 (200k showers / 5.0M rows), so this is 120x, and it
is the generator OVER-CONCENTRATING, not the under-merging first written down. See
experiment-memory/2026-09-15-calo-generator-OVER-concentrates-corrects-under-merge.md.

THE HYPOTHESIS: the excess sits in the shower CORE. A shower's core is where cells are densest, so a
point cloud whose central density is too high collides there first. `PointCFM` controls that density
directly, which would make the defect testable by resampling rather than retraining.

THE ALTERNATIVE worth keeping in view: the excess is uniform in radius, which would instead mean the
cloud is globally too tight (a width problem, not a core-shape problem) -- and `shower_width`
W/sigma is only 0.0505, so a pure width error is not indicated and a uniform result would be
genuinely surprising.

WHAT IS MEASURED
  A. real reference (free, no model). Within-shower nearest-neighbour distance between CELLS, and
     the cell-count radial profile. Real cells cannot collide -- they ARE distinct cells -- so the
     reference is the SPACING the generator has to reproduce, not a collision rate.
  B. generated. Same statistics from sampled showers, plus the collision rate binned by distance
     from the shower core in units of the shower width.
  C. the comparison that decides it: collision rate vs r/width. Rising sharply toward r=0 means the
     core; flat means a global density error.

Usage:
  python scripts/calo_cell_cooccupancy.py --ckpt <ckpt.pt> --real_slice <slice.npz> --n_showers 40000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from genpu.calo_cells import cell_ids, etaphidepth_to_xyz, snap_cells  # noqa: E402
from genpu.flow.calo_flow import CaloFlow  # noqa: E402


def within_shower_nn(d_eta, d_phi, src, n_max=400, rng=None):
    """Median nearest-neighbour distance between cells of the same shower, per shower.

    Showers are capped at `n_max` cells for the O(n^2); the cap is reported so a reader can see
    whether it bound anything. Distances are in (d_eta, d_phi) units, which is the frame both the
    real slice and the sampler use -- converting to mm would need a per-cell radius and would mix
    the barrel and endcap scales.
    """
    rng = rng or np.random.default_rng(0)
    order = np.argsort(src, kind="stable")
    s = src[order]; de = d_eta[order]; dp = d_phi[order]
    bounds = np.searchsorted(s, np.arange(s[-1] + 2))
    out = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 2:
            continue
        idx = np.arange(a, b)
        if b - a > n_max:
            idx = rng.choice(idx, n_max, replace=False)
        P = np.stack([de[idx], dp[idx]], 1)
        d = np.abs(P[:, None, :] - P[None, :, :]).max(-1)
        np.fill_diagonal(d, np.inf)
        out.append(np.median(d.min(1)))
    return np.array(out)


def collision_profile(r_over_w, collided, edges):
    """Collision rate as a function of distance from the core, in units of the shower width."""
    idx = np.clip(np.digitize(r_over_w, edges) - 1, 0, len(edges) - 2)
    num = np.bincount(idx, weights=collided.astype(float), minlength=len(edges) - 1)
    den = np.bincount(idx, minlength=len(edges) - 1)
    return num / np.maximum(den, 1), den.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real_slice", required=True)
    ap.add_argument("--n_showers", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--max_cells", type=int, default=4096)
    ap.add_argument("--nn_showers", type=int, default=4000, help="showers used for the O(n^2) NN stat")
    ap.add_argument("--out", default="calo_cooccupancy.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.real_slice, allow_pickle=True)
    S = min(a.n_showers, len(z["offsets"]) - 1)
    off = z["offsets"]; pts = z["points_flat"]; n = np.diff(off)[:S]; lim = off[S]
    src = np.repeat(np.arange(S), n)
    # RAW cont -- the slice stores it unstandardised and the model standardises internally. A probe
    # that re-applied (x*std+mean) here inflated |eta| to a median of 4.5 and put 83% of cells inside
    # the beam pipe; `calo_metrics.py:172` does it correctly and so must this.
    p_eta = z["cont"][:S, 1].astype(np.float64); p_phi = z["p_phi"][:S].astype(np.float64)
    core = (z["glob"][:S, 2:4] + z["anchor"][:S]).astype(np.float64)
    out = {"n_showers": int(S), "ckpt": a.ckpt}

    # ---- A. real reference -------------------------------------------------
    r_eta = p_eta[src] + core[src, 0] + pts[:lim, 0]
    r_phi = p_phi[src] + core[src, 1] + pts[:lim, 1]
    rx, ry, rz, rdet = etaphidepth_to_xyz(r_eta, r_phi, pts[:lim, 2])
    rxs, rys, rzs, _ = snap_cells(rx, ry, rz, rdet)
    rcid = cell_ids(rxs, rys, rzs, rdet)
    u, cnt = np.unique(np.stack([rcid, src], 1), axis=0, return_counts=True)
    real_coll = 1 - len(u) / lim
    # PHYSICAL SANITY CHECK, run before anything else is believed: an endcap cell cannot sit inside
    # the beam pipe. ecal_e_inner_radius = 315 mm.
    me = np.isin(rdet, (9, 11))
    frac_inside = float(np.mean(np.hypot(rx[me], ry[me]) < 315.0)) if me.any() else 0.0
    print(f"REAL: {lim:,} cells / {S:,} showers   collision {real_coll:.4f}   "
          f"endcap cells inside r=315mm {frac_inside:.4f} (must be ~0)")
    k = min(a.nn_showers, S)
    mk = src < k
    real_nn = within_shower_nn(pts[:lim, 0][mk], pts[:lim, 1][mk], src[mk])
    print(f"  within-shower NN spacing: median {np.median(real_nn):.5f}  "
          f"p10 {np.percentile(real_nn,10):.5f}  p90 {np.percentile(real_nn,90):.5f}")
    out["real"] = {"collision": round(float(real_coll), 5), "frac_inside_beampipe": frac_inside,
                   "nn_med": float(np.median(real_nn)),
                   "nn_p10": float(np.percentile(real_nn, 10)),
                   "nn_p90": float(np.percentile(real_nn, 90))}

    # ---- B. generated ------------------------------------------------------
    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = sd.get("model", sd)
    norm = {kk: z[kk] for kk in ("cont_mean", "cont_std", "glob_mean", "glob_std", "pts_mean", "pts_std")}
    model, _, _ = CaloFlow.from_checkpoint(sd, norm)
    model.to(dev).eval()
    contS = torch.as_tensor((z["cont"][:S] - norm["cont_mean"]) / norm["cont_std"],
                            dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(z["pdg"][:S].astype(np.int64), device=dev)
    ancT = torch.as_tensor(z["anchor"][:S], dtype=torch.float32, device=dev) if bool(model.core_anchored > 0) else None
    amT = torch.as_tensor(z["anchor_mode"][:S].astype(np.int64), device=dev) if model.anchor_cond else None
    g_de, g_dp, g_dep, g_src = [], [], [], []
    for s0 in range(0, S, a.batch):
        e0 = min(s0 + a.batch, S)
        with torch.no_grad():
            sh = model.sample_showers(contS[s0:e0], pdgT[s0:e0], steps=a.steps, max_cells=a.max_cells,
                                      core_anchor=None if ancT is None else ancT[s0:e0],
                                      anchor_mode=None if amT is None else amT[s0:e0])
        rp = sh["src"].cpu().numpy(); pos = sh["pos"].cpu().numpy(); co = sh["core"].cpu().numpy()
        g_de.append(co[rp, 0] + pos[:, 0]); g_dp.append(co[rp, 1] + pos[:, 1])
        g_dep.append(pos[:, 2]); g_src.append(rp + s0)
    g_de = np.concatenate(g_de); g_dp = np.concatenate(g_dp)
    g_dep = np.concatenate(g_dep); g_src = np.concatenate(g_src)
    gx, gy, gz, gdet = etaphidepth_to_xyz(p_eta[g_src] + g_de, p_phi[g_src] + g_dp, g_dep)
    gxs, gys, gzs, _ = snap_cells(gx, gy, gz, gdet)
    gcid = cell_ids(gxs, gys, gzs, gdet)
    uu, inv, cc = np.unique(np.stack([gcid, g_src], 1), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel(); collided = cc[inv] > 1
    gen_coll = 1 - len(uu) / len(gcid)
    print(f"GEN : {len(gcid):,} points / {S:,} showers   collision {gen_coll:.4f}   "
          f"[real {real_coll:.4f}]")

    # ---- C. collisions vs distance from the core ---------------------------
    # r is measured from the shower's own core (pos is already core-relative) and normalised by the
    # shower's rms width, so the profile is comparable across showers of very different size.
    # g_de/g_dp are core-INCLUSIVE (core + pos), so subtract the core back out to get the
    # core-relative radius.
    dr = np.hypot(g_de - core[g_src, 0], g_dp - core[g_src, 1])
    w = np.sqrt(np.bincount(g_src, weights=dr ** 2, minlength=S) / np.maximum(np.bincount(g_src, minlength=S), 1))
    r_over_w = dr / np.maximum(w[g_src], 1e-9)
    edges = np.array([0, .25, .5, .75, 1., 1.5, 2., 3., 5., 100.])
    prof, dens = collision_profile(r_over_w, collided, edges)
    print("\ncollision rate vs distance from the shower core (units of shower rms width):")
    print(f"{'r/width':>12} {'points':>10} {'collision':>10}")
    for i in range(len(edges) - 1):
        print(f"{edges[i]:>5.2f}-{edges[i+1]:<6.2f} {dens[i]:>10,} {prof[i]:>10.4f}")
    out["gen"] = {"collision": round(float(gen_coll), 5),
                  "profile_edges": edges.tolist(),
                  "profile_rate": [round(float(v), 5) for v in prof],
                  "profile_n": dens.tolist()}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    print("READ THE PROFILE: rising sharply toward r=0 means the CORE is too dense, which PointCFM "
          "controls directly and which is testable by resampling. Flat means a global density error.")


if __name__ == "__main__":
    main()
