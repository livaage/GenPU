"""Phase 0b DECISION GATE: how much of the shower core does a truth-helix extrapolation explain?

The core (shower centroid offset from the particle direction) carries 83-93% of the "shower width"
variance the gate reacts to, and is a 1.5 m magnetic-bending displacement (pion median |core| 1.54
rad) that we currently ask a Gaussian mixture to predict from vertex kinematics. If instead we
extrapolate the particle's helix to the calo front face and store the core as a RESIDUAL from that
prediction, the mixture's job becomes small and local — the same frame-change that gave v3
surface-local and the v4 helix anchor their wins.

This measures the frame change WITHOUT training anything:
    spread( core relative to the PARTICLE direction )   <- what the model predicts today
    spread( core relative to the HELIX prediction   )   <- what it would predict after the change
GO if the residual is materially tighter (target >~2x for pions).

Uses truth conditioning only (pT, eta, phi, charge, vertex) — no generated quantities.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# the extrapolation itself now lives in genpu.calo_geom, so the probe, the slice builder and the
# generation path share ONE implementation of the anchor (Phase 1 depends on them agreeing exactly)
from genpu.calo_geom import helix_to_face, wrap_pi, MODE_NONE

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)


def spread(a):
    q1, q3 = np.percentile(a, [25, 75])
    return (q3 - q1) / 1.349, np.std(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--geometry", default="/home/lv7805/genpu/calo_geometry.json")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--tag", default="pion")
    ap.add_argument("--min_cells", type=int, default=2)
    ap.add_argument("--max_showers", type=int, default=400000)
    ap.add_argument("--anchor_kind", choices=["helix", "line", "auto"], default="helix",
                    help="'line' forces the straight-line path for charged particles too — the "
                         "bremsstrahlung hypothesis (radiated photons travel straight from the "
                         "radiation point, so a stiff e±'s own curved path is the wrong predictor)")
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics")
    args = ap.parse_args()

    g = json.loads(Path(args.geometry).read_text())["front_face"]
    R, Z = g["barrel_r"], g["endcap_absz"]
    print(f"[{args.tag}] calo front face: barrel r={R:.0f} mm, endcap |z|={Z:.0f} mm, "
          f"eta transition {g['eta_transition']:.3f}")

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, ch, off = d["particle_features"], d["particle_aux"], d["calo_hits_flat"], d["calo_offsets"]
    ncell = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ncell >= args.min_cells))[0]
    if len(sel) > args.max_showers:
        sel = sel[:args.max_showers]
    print(f"  {len(sel)} showers with >= {args.min_cells} cells")

    pt = np.exp(pf[sel, PF_LOGPT]).astype(np.float64)
    p_eta, p_phi = pf[sel, PF_ETA].astype(np.float64), pf[sel, PF_PHI].astype(np.float64)
    q = pf[sel, PF_CHARGE].astype(np.float64)
    vx, vy, vz = (aux[sel, AUX_VX].astype(np.float64), aux[sel, AUX_VY].astype(np.float64),
                  aux[sel, AUX_VZ].astype(np.float64))

    h_eta, h_phi, amode = helix_to_face(pt, p_phi, p_eta, q, vx, vy, vz, R, Z, kind=args.anchor_kind)
    ok = amode != MODE_NONE
    print(f"  anchor branches: barrel {(amode==0).mean():.3f}  endcap {(amode==1).mean():.3f}  "
          f"turning point {(amode==2).mean():.3f}  none {(amode==3).mean():.3f}")

    # cores measured the same way as the slice builder (mean of wrapped per-cell deltas), once
    # relative to the PARTICLE direction and once relative to the HELIX prediction
    core_p_eta = np.zeros(len(sel)); core_p_phi = np.zeros(len(sel))
    core_h_eta = np.zeros(len(sel)); core_h_phi = np.zeros(len(sel))
    for k, i in enumerate(sel):
        a, b = off[i], off[i + 1]
        ce, cp = ch[a:b, CH_ETA].astype(np.float64), ch[a:b, CH_PHI].astype(np.float64)
        core_p_eta[k] = np.mean(ce - p_eta[k]); core_p_phi[k] = np.mean(wrap_pi(cp - p_phi[k]))
        core_h_eta[k] = np.mean(ce - h_eta[k]); core_h_phi[k] = np.mean(wrap_pi(cp - h_phi[k]))

    m = ok
    print(f"\n  {'anchor':>10} {'|core| med':>11} {'phi sigma':>10} {'eta sigma':>10} {'phi IQR/1.349':>14}")
    res = {}
    for lab, ce_, cp_ in [("particle", core_p_eta[m], core_p_phi[m]),
                          (args.anchor_kind, core_h_eta[m], core_h_phi[m])]:
        s_phi, sd_phi = spread(cp_); s_eta, _ = spread(ce_)
        mag = np.median(np.hypot(ce_, cp_))
        print(f"  {lab:>10} {mag:>11.4f} {sd_phi:>10.4f} {np.std(ce_):>10.4f} {s_phi:>14.4f}")
        res[lab] = {"core_mag_median": float(mag), "phi_std": float(sd_phi), "eta_std": float(np.std(ce_)),
                    "phi_robust_sigma": float(s_phi), "eta_robust_sigma": float(s_eta)}
    imp_phi = res["particle"]["phi_robust_sigma"] / max(res[args.anchor_kind]["phi_robust_sigma"], 1e-9)
    imp_eta = res["particle"]["eta_robust_sigma"] / max(res[args.anchor_kind]["eta_robust_sigma"], 1e-9)
    print(f"\n  TIGHTENING (particle -> {args.anchor_kind} anchor):  phi {imp_phi:.2f}x   eta {imp_eta:.2f}x")

    # per-pT breakdown: bending is a 1/pT effect, so the gain should be largest at low pT
    edges = np.quantile(np.log(pt[m]), np.linspace(0, 1, 7))
    bi = np.clip(np.digitize(np.log(pt[m]), edges[1:-1]), 0, 5)
    print(f"\n  {'pT bin':>7} {'pT med':>8} {'n':>8} {'sig_phi particle':>17} {'sig_phi helix':>14} {'gain':>6}")
    for b in range(6):
        s = bi == b
        if s.sum() < 200:
            continue
        a1, _ = spread(core_p_phi[m][s]); a2, _ = spread(core_h_phi[m][s])
        print(f"  {b:>7} {np.median(pt[m][s]):>8.3f} {s.sum():>8} {a1:>17.4f} {a2:>14.4f} {a1/max(a2,1e-9):>6.2f}x")

    # per-BRANCH: the face crossing and the curler turning point are different physics claims and
    # cover different populations, so a pooled number would hide either one failing
    print(f"\n  {'branch':>14} {'n':>8} {'sig_phi particle':>17} {'sig_phi anchor':>15} {'gain':>6}")
    per_mode = {}
    for code, lab in [(0, "barrel"), (1, "endcap"), (2, "turning pt")]:
        s = (amode == code) & m
        if s.sum() < 200:
            continue
        a1, _ = spread(core_p_phi[s]); a2, _ = spread(core_h_phi[s])
        print(f"  {lab:>14} {s.sum():>8} {a1:>17.4f} {a2:>15.4f} {a1/max(a2,1e-9):>6.2f}x")
        per_mode[lab] = {"n": int(s.sum()), "sigma_phi_particle": float(a1),
                         "sigma_phi_anchor": float(a2), "gain": float(a1 / max(a2, 1e-9))}
    res["per_branch"] = per_mode
    res["branch_frac"] = {lab: float((amode == c).mean())
                          for c, lab in [(0, "barrel"), (1, "endcap"), (2, "turning"), (3, "none")]}

    verdict = ("GO — helix anchor materially tightens the core" if imp_phi >= 2.0 else
               "MARGINAL — some tightening, weigh against complexity" if imp_phi >= 1.3 else
               "NO-GO — the helix does not explain this species' core")
    print(f"\n  VERDICT ({args.tag}): {verdict}")
    res.update({"tag": args.tag, "anchor_kind": args.anchor_kind,
                "valid_frac": float(ok.mean()), "tighten_phi": float(imp_phi),
                "tighten_eta": float(imp_eta), "verdict": verdict, "n": int(m.sum())})
    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    (o / f"helix_core_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print("  wrote", o / f"helix_core_{args.tag}.json")


if __name__ == "__main__":
    main()
