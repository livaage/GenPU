"""How hard does `partition` rescale, and would it SMEAR a truncated energy edge?

The floor redesign (2026-08-27) is to drop `EnergyHead`'s at-floor point mass -- which puts ~0.56%
of generated cells at exactly log(5e-5), a value real data never contains -- and replace it with a
mixture TRUNCATED at log(5e-5), the real hard edge (`calo_hits.total_energy` min>0 = 5.0001e-05,
frac below = 0.00000).

But `sample_showers` renormalises AFTER the draw: above-floor cells are multiplied by

    s_rest = (total - sum_floor) / sum_rest

so a truncated edge at `log_floor` lands at `log_floor + log(s_rest)` -- a DIFFERENT place in every
shower. That is the mechanism, and this measures its size:

  |log s_rest| small  -> the edge barely moves; truncation is worth doing on its own, and the
                         simplex-fraction reparameterisation can follow as a separate arm.
  |log s_rest| large  -> truncation is pointless until energies are generated as FRACTIONS on the
                         simplex (sum exact by construction, no post-hoc rescale, edge stays put).

Method: sample with `partition=False`, which returns the GlobalHead's `total` alongside the raw
per-cell draws, then recompute exactly what `partition` would have done (`calo_flow.py`, the
`if partition:` block) without applying it. Nothing is approximated -- the same `is_floor`,
`sum_floor`, `sum_rest`, `resid`, `ok` and `scale_bounds` logic is reproduced verbatim.

Reported in log units because that is what moves the edge: the edge shift IS `log(s_rest)`, and the
comparison scale is the real density step at the threshold (~17x over 0.1 in log-E, i.e. a 0.1-wide
smear already blurs it materially).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--real_slice", required=True)
    ap.add_argument("--pdg_class", type=int, nargs="+", default=None)
    ap.add_argument("--max_showers", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--scale_bounds", type=float, nargs=2, default=(1e-2, 1e2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    a = ap.parse_args()
    import torch
    from genpu.flow.calo_flow import CaloFlow
    rng = np.random.default_rng(a.seed); t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = ck["model"]; ns = ck.get("norm", sd)

    def _get(b):
        for k in (f"cont_{b}", f"cond_{b}"):
            if k in ns:
                v = ns[k]; return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        raise KeyError(b)
    norm = {"cont_mean": _get("mean"), "cont_std": _get("std")}
    for k in ["glob_mean", "glob_std", "pts_mean", "pts_std"]:
        v = ns[k]; norm[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    model, _, _ = CaloFlow.from_checkpoint(sd, norm); model = model.to(dev).eval()
    log_floor = float(model.log_floor)

    d = np.load(a.real_slice)
    pdg_all = d["pdg"].astype(np.int64)
    sel = (np.arange(len(pdg_all)) if a.pdg_class is None
           else np.where(np.isin(pdg_all, a.pdg_class))[0])
    if len(sel) > a.max_showers:
        sel = np.sort(rng.choice(sel, a.max_showers, replace=False))
    cont = d["cont"].astype(np.float32)[sel]
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg_all[sel], device=dev)
    E_trueT = torch.as_tensor(d["E_true"].astype(np.float32)[sel], device=dev)
    aT = (torch.as_tensor(d["anchor"].astype(np.float32)[sel], device=dev)
          if bool(model.core_anchored > 0) else None)
    mT = (torch.as_tensor(d["anchor_mode"].astype(np.int64)[sel], device=dev)
          if model.anchor_cond else None)
    print(f"{len(sel):,} showers  log_floor={log_floor:.4f}  dev={dev}", flush=True)

    S_REST, S_ALL, OKF, NPER, FFLOOR = [], [], [], [], []
    for s in range(0, len(sel), a.batch):
        e = min(s + a.batch, len(sel))
        with torch.no_grad():
            # partition=False -> raw per-cell draws PLUS the GlobalHead total, which is exactly
            # the pair `partition` combines. e_true stays off so the energy-conservation shrink
            # (a separate, later step) cannot contaminate the scale.
            sh = model.sample_showers(contS[s:e], pdgT[s:e], steps=a.steps, partition=False,
                                      e_true=None,
                                      core_anchor=None if aT is None else aT[s:e],
                                      anchor_mode=None if mT is None else mT[s:e])
        logE = sh["logE"]; src = sh["src"]; total = sh["total"]; n = sh["n"]
        S = total.shape[0]
        en = torch.exp(logE)
        is_floor = logE <= model.log_floor + 1e-6
        sum_floor = torch.zeros(S, device=en.device).index_add_(0, src, torch.where(is_floor, en, torch.zeros_like(en)))
        sum_rest = torch.zeros(S, device=en.device).index_add_(0, src, torch.where(is_floor, torch.zeros_like(en), en))
        resid = total - sum_floor
        ok = (resid > 0) & (sum_rest > 0)
        s_rest = torch.where(ok, resid / sum_rest.clamp(min=1e-30), torch.ones_like(resid))
        s_all = torch.where(ok, torch.ones_like(resid), total / (sum_floor + sum_rest).clamp(min=1e-30))
        ffl = torch.zeros(S, device=en.device).index_add_(0, src, is_floor.float()) / n.to(en.dtype)
        for L, v in ((S_REST, s_rest), (S_ALL, s_all), (OKF, ok.float()), (NPER, n), (FFLOOR, ffl)):
            L.append(v.cpu().numpy())
    s_rest = np.concatenate(S_REST).astype(np.float64)
    s_all = np.concatenate(S_ALL).astype(np.float64)
    okf = np.concatenate(OKF).astype(bool)
    n_per = np.concatenate(NPER).astype(np.int64)
    ffloor = np.concatenate(FFLOOR).astype(np.float64)
    lo, hi = a.scale_bounds
    sat = (s_rest < lo) | (s_rest > hi) | (s_all < lo) | (s_all > hi)

    ls = np.log(np.clip(s_rest[okf], 1e-30, None))
    q = lambda x, p: float(np.percentile(x, p))
    out = {"ckpt": a.ckpt, "showers": int(len(s_rest)), "log_floor": log_floor,
           "frac_ok": float(okf.mean()), "frac_saturated": float(sat.mean()),
           "frac_at_floor_mean": float(ffloor.mean()),
           "log_s_rest": {p: q(ls, p) for p in (1, 5, 25, 50, 75, 95, 99)},
           "abs_log_s_rest": {"mean": float(np.abs(ls).mean()),
                              "median": float(np.median(np.abs(ls))),
                              "p90": q(np.abs(ls), 90), "p99": q(np.abs(ls), 99)},
           "frac_shift_gt_0p1": float((np.abs(ls) > 0.1).mean()),
           "frac_shift_gt_0p5": float((np.abs(ls) > 0.5).mean())}

    print(f"\npartition applies the normal branch to {100*okf.mean():.2f}% of showers "
          f"(saturated {100*sat.mean():.3f}%); mean generated at-floor fraction {ffloor.mean():.4f}")
    print(f"\nEDGE SHIFT = log(s_rest), in log-E units:")
    for p in (1, 5, 25, 50, 75, 95, 99):
        print(f"    p{p:<3d} {out['log_s_rest'][p]:+8.4f}")
    print(f"\n    mean|shift| {out['abs_log_s_rest']['mean']:.4f}   median {out['abs_log_s_rest']['median']:.4f}"
          f"   p90 {out['abs_log_s_rest']['p90']:.4f}   p99 {out['abs_log_s_rest']['p99']:.4f}")
    print(f"    showers whose edge moves >0.1 in log-E: {100*out['frac_shift_gt_0p1']:.2f}%")
    print(f"    showers whose edge moves >0.5 in log-E: {100*out['frac_shift_gt_0p5']:.2f}%")
    print(f"\nREADING: the real density step at the threshold is ~17x across 0.1 in log-E, so a"
          f"\nsmear comparable to 0.1 already blurs the edge this redesign exists to reproduce.")

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / f"partition_scale_{a.tag}.json"
    jp.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {jp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
