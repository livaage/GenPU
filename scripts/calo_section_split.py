"""How much of rho(logE, depth) is SHOWER PHYSICS, and how much is the ECAL->HCAL SAMPLING STEP?

2026-08-27 measured a within-shower energy/depth gradient the model reproduces none of, peaking at
rho = +0.542 for protons, and read it as the longitudinal shower profile. But the calorimeter is two
detectors, and the slice's depth coordinate cannot tell them apart:

  region              det      depth p1..p99      <logE>     layer pitch
  barrel ECAL          10       -6 ..  228        -7.97       5.050 mm
  endcap ECAL         9,11      -10 ..  227       -8.07       5.050 mm
  barrel HCAL          13      388 .. 1446        -6.86      51.000 mm
  endcap HCAL        12,14     435 .. 2220        -6.89      51.000 mm

Mean cell energy STEPS by ~3x across the gap, because an HCAL cell integrates ~10x more material.
Deep cells are hot for a reason that has nothing to do with the shower. Since deep-penetrating
species (p, K, pi) put more of their cells past the boundary, a large part of the measured gradient
may be this step rather than physics -- and the two need opposite treatment: a sampling step is
deterministic geometry that should be GIVEN to the model, a shower profile is physics it must learn.

TEST. Recompute the same within-shower Spearman rho(logE, depth) three ways:
  all      every cell (reproduces the 2026-08-27 number as a sanity check)
  ECAL     restricted to each shower's ECAL cells
  HCAL     restricted to each shower's HCAL cells
  step-rm  every cell, but log-E centred WITHIN ITS SECTION first -- removes exactly the step and
           nothing else, so it isolates the within-section gradient over the full depth range

If rho collapses in the section-restricted and step-removed variants, the headline was geometry.
If it survives, it was physics. Both can be partly true and the split is the answer.

SECTION ASSIGNMENT is exact enough to be safe. `point_layer >= 0` marks endcap cells exactly (the
builder sets a layer only for dets 9/11/12/14). Depth then separates the sections with a wide
guard band over the physical gap, and cells inside the band are DROPPED and counted rather than
guessed at.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NAME = {0: "e-", 1: "e+", 2: "gamma", 3: "pi+", 4: "pi-", 5: "K+", 6: "K-", 7: "p", 8: "pbar",
        9: "n", 10: "nbar", 11: "mu-", 12: "mu+", 13: "K0L", 14: "K0S", 15: "pi0", 16: "other"}
# measured from calo_geometry.json + raw calo_hits, 2026-08-27 (see module docstring)
ECAL_MAX, HCAL_MIN = 250.0, 380.0


def within_rank(vals, off, n_per, rng, jitter=0.0):
    v = vals + (rng.normal(0, jitter, len(vals)) if jitter else 0.0)
    src = np.repeat(np.arange(len(n_per)), n_per)
    order = np.lexsort((v, src))
    r = np.empty(len(v), np.float64)
    r[order] = np.arange(len(v), dtype=np.float64) - off[src[order]]
    out = r / np.maximum(n_per - 1, 1)[src]
    out[np.repeat(n_per == 1, n_per)] = 0.5
    return out


def within_spearman(x, y, off, n_per, rng, min_n=3):
    """Mean over showers of the within-shower Spearman rho(x, y)."""
    if len(n_per) == 0 or n_per.sum() == 0:
        return float("nan"), float("nan"), 0
    rx = within_rank(x, off, n_per, rng)
    ry = within_rank(y, off, n_per, rng, jitter=1e-4)
    src = np.repeat(np.arange(len(n_per)), n_per)
    S = len(n_per)
    mx = np.bincount(src, weights=rx, minlength=S) / n_per
    my = np.bincount(src, weights=ry, minlength=S) / n_per
    dx, dy = rx - mx[src], ry - my[src]
    num = np.bincount(src, weights=dx * dy, minlength=S)
    den = np.sqrt(np.bincount(src, weights=dx * dx, minlength=S)
                  * np.bincount(src, weights=dy * dy, minlength=S))
    keep = n_per >= min_n
    rho = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)[keep]
    if not len(rho):
        return float("nan"), float("nan"), 0
    return float(rho.mean()), float(rho.std() / np.sqrt(len(rho))), int(len(rho))


def compact(mask, off, n_per):
    """Restrict a CSR set of showers to the cells selected by `mask`; returns (idx, off2, n2)."""
    n2 = np.bincount(np.repeat(np.arange(len(n_per)), n_per)[mask], minlength=len(n_per))
    idx = np.where(mask)[0]
    return idx, np.concatenate([[0], np.cumsum(n2)]).astype(np.int64), n2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_slice", required=True)
    ap.add_argument("--max_showers", type=int, default=200000)
    ap.add_argument("--min_n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); t0 = time.time()

    d = np.load(a.real_slice)
    pts, off_all = d["points_flat"], d["offsets"].astype(np.int64)
    pdg = d["pdg"].astype(np.int64)
    lay = d["point_layer"] if "point_layer" in d.files else None
    if pts.shape[1] < 4:
        raise SystemExit("v1 slice: no depth column.")
    if lay is None:
        raise SystemExit("slice has no point_layer; cannot separate barrel from endcap exactly.")
    n_all = np.diff(off_all)
    print(f"slice {Path(a.real_slice).name}: {len(pdg):,} showers, {len(pts):,} cells", flush=True)

    res = {"slice": a.real_slice, "ECAL_MAX": ECAL_MAX, "HCAL_MIN": HCAL_MIN, "classes": {}}
    print(f"\n{'class':7s} {'showers':>9s} {'%ECAL':>6s} {'%HCAL':>6s} {'%band':>6s} {'step':>7s} | "
          f"{'rho ALL':>9s} {'rho ECAL':>9s} {'rho HCAL':>9s} {'rho step-rm':>11s}")
    print("-" * 96)
    for c in [None] + sorted(set(pdg.tolist())):
        sel = np.arange(len(pdg)) if c is None else np.where(pdg == c)[0]
        if len(sel) < 2000:
            continue
        if len(sel) > a.max_showers:
            sel = np.sort(rng.choice(sel, a.max_showers, replace=False))
        n_per = n_all[sel]
        ci = (np.repeat(off_all[sel], n_per)
              + (np.arange(n_per.sum()) - np.repeat(np.cumsum(n_per) - n_per, n_per)))
        dep = pts[ci, 2].astype(np.float64)
        le = pts[ci, 3].astype(np.float64)
        off = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)

        is_ecal, is_hcal = dep < ECAL_MAX, dep > HCAL_MIN
        band = ~(is_ecal | is_hcal)
        fE, fH, fB = is_ecal.mean(), is_hcal.mean(), band.mean()
        step = (le[is_hcal].mean() - le[is_ecal].mean()) if (is_hcal.any() and is_ecal.any()) else np.nan

        rho_all, se_all, ns = within_spearman(dep, le, off, n_per, rng, a.min_n)
        out = {"showers": int(len(sel)), "frac_ECAL": float(fE), "frac_HCAL": float(fH),
               "frac_band": float(fB), "step_logE": float(step),
               "rho_all": rho_all, "sem_all": se_all, "n_showers_all": ns}
        for lab, m in (("ECAL", is_ecal), ("HCAL", is_hcal)):
            idx, off2, n2 = compact(m, off, n_per)
            r_, s_, k_ = within_spearman(dep[idx], le[idx], off2, n2, rng, a.min_n)
            out[f"rho_{lab}"], out[f"sem_{lab}"], out[f"n_showers_{lab}"] = r_, s_, k_
        # step removed: centre log-E within its own section, keep every cell and the full depth range
        le_c = le.copy()
        for m in (is_ecal, is_hcal):
            if m.any():
                le_c[m] -= le[m].mean()
        if band.any():
            le_c[band] -= le[band].mean()
        r_s, s_s, k_s = within_spearman(dep, le_c, off, n_per, rng, a.min_n)
        out["rho_steprm"], out["sem_steprm"] = r_s, s_s
        res["classes"]["ALL" if c is None else NAME.get(c, str(c))] = out

        lab = "ALL" if c is None else NAME.get(c, str(c))
        print(f"{lab:7s} {len(sel):9,d} {100*fE:5.1f}% {100*fH:5.1f}% {100*fB:5.1f}% {step:+7.3f} | "
              f"{rho_all:+9.4f} {out['rho_ECAL']:+9.4f} {out['rho_HCAL']:+9.4f} {r_s:+11.4f}",
              flush=True)

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / f"section_split_{a.tag}.json"
    jp.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {jp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
