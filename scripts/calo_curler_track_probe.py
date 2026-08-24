"""Do CURLERS leave enough tracker hits for a track endpoint to be worth conditioning on?

63% of pion / 72% of e± calo showers are anchored at the helix TURNING POINT: the particle is
soft (median pT 0.27 GeV), born deep in the tracker (vr ~ 420 mm), and curls back before ever
reaching the calo face. Its calo cells are decay/secondary energy booked to the parent. The
turning-point anchor is worth ~2.5x (vs 15-24x on the face branch), and it is a PROXY — computed
from the VACUUM helix, so it knows nothing about dE/dx, multiple scattering, nuclear interaction
or early stopping.

The proposed fix is to anchor on the generated TRACK's endpoint instead. That is only worth
building if the tracker actually sees these particles. This probe answers three questions on real
data, with no model and no training:

  Q1  COVERAGE  — how many tracker hits does each anchor branch leave? A particle with 0 hits
                  gets no track edge at all; with 1-2 hits the endpoint is nearly uninformative.
  Q2  ANCHOR ERROR — how far is the vacuum-helix turning point from where the track ACTUALLY got
                  to? This is the information the helix cannot have, in mm and in rad.
  Q3  THE MONEY NUMBER — does the track endpoint predict the shower core BETTER than the helix
                  anchor does? Phase 0b's methodology, with the real endpoint substituted for the
                  helix prediction: sigma(core - anchor_track) vs sigma(core - anchor_helix), per
                  branch. If this is not materially tighter, the track edge is not worth building
                  and the turning-point proxy is as good as it gets.

IMPORTANT ordering subtlety: we want the OUTERMOST hit (the empirical analogue of the helix
turning point) — "how far out did it get", not "where did it stop in time".

CORRECTED 2026-08-24: this probe previously took the LAST STORED hit, on the CLAUDE.md claim that
stage2 hits are r-ascending. They are NOT — stage2 order is arbitrary (Spearman(index, r) = 0.01,
median max r-drop 158 mm); the r-sort lives in build_tracker_slice.py, not preprocessing. So the
old Q2/Q3 numbers were computed on an essentially RANDOM hit. Q1 (coverage = hit COUNTS) is
unaffected, which is the Phase 4 headline. We now take argmax(r) explicitly.

Real data only (truth kinematics + real tracker hits + real calo cells): this is a measurement of
what information EXISTS, upstream of any exposure-bias question about generated tracks.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.calo_geom import core_anchor, load_front_face, wrap_pi  # noqa: E402

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)

BRANCH = {0: "barrel", 1: "endcap", 2: "turning", 3: "none"}
# dataviz reference palette, categorical slots 1-3 in fixed order (light mode)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def csr_last(offsets, sel, r_col):
    """Index of the OUTERMOST hit (max r) per selected particle, and the hit count.

    Must be computed, not assumed: stage2 hit order is arbitrary (see the module docstring).
    `r_col` is the r column of the flat hit array.
    """
    n = (offsets[sel + 1] - offsets[sel]).astype(np.int64)
    out = np.full(len(sel), -1, dtype=np.int64)
    for i, (p, cnt) in enumerate(zip(sel, n)):
        if cnt > 0:
            lo = int(offsets[p])
            out[i] = lo + int(np.argmax(r_col[lo:lo + cnt]))
    return out, n


def csr_gather(offsets, sel):
    n = (offsets[sel + 1] - offsets[sel]).astype(np.int64)
    tot = int(n.sum())
    owner = np.repeat(np.arange(len(sel), dtype=np.int64), n)
    within = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    return np.repeat(offsets[sel].astype(np.int64), n) + within, owner, n


def q(a, name, fmt="{:.1f}"):
    if len(a) == 0:
        return f"{name}: (empty)"
    p = np.percentile(a, [10, 25, 50, 75, 90])
    return f"{name}: " + " ".join(fmt.format(x) for x in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4],
                    help="3 4 = pi+/pi-, 0 1 = e-/e+")
    ap.add_argument("--shard", type=int, default=5)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--tag", default="pion")
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--born_outside_only", action="store_true",
                    help="Q3 CLEAN (2026-08-24): keep only depositors born OUTSIDE the calo front "
                         "face. The turning-point branch is 81%% (pion) / 91%% (e±) particles created "
                         "INSIDE the calorimeter -- endcap shower fragments at vr ~ 420 mm but "
                         "|vz| > 3212 mm -- which have zero tracker hits by construction, never "
                         "having been in the tracker. Pooling them is what produced the 2026-08-16 "
                         "'81/93%% of curlers are invisible' number and the Phase 4 falsification. "
                         "For genuine tracker-born curlers 95.5%% (pion) / ~77%% (e±) DO leave hits, "
                         "so this flag measures the population the track endpoint could actually "
                         "serve. Job 12874782.")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux = d["particle_features"], d["particle_aux"]
    coff, toff = d["calo_offsets"], d["tracker_offsets"]
    ncal = np.diff(coff)
    keep = np.isin(pf[:, PF_PDG], args.pdg_class) & (ncal >= 1)
    if args.born_outside_only:
        _R, _Z = load_front_face(args.geometry)
        _vr = np.hypot(aux[:, AUX_VX], aux[:, AUX_VY])
        born_out = (_vr < _R) & (np.abs(aux[:, AUX_VZ]) < _Z)
        print(f"born_outside_only: {keep.sum():,} depositors -> {(keep & born_out).sum():,} "
              f"({(born_out[keep]).mean():.3f} kept)")
        keep &= born_out
    sel = np.where(keep)[0]
    if len(sel) > args.n:
        sel = np.sort(rng.choice(sel, args.n, replace=False))
    print(f"shard {args.shard}: {len(sel)} showers, pdg_class {args.pdg_class}")

    pt = np.exp(pf[sel, PF_LOGPT]).astype(np.float64)
    p_eta, p_phi = pf[sel, PF_ETA].astype(np.float64), pf[sel, PF_PHI].astype(np.float64)
    qch = pf[sel, PF_CHARGE].astype(np.float64)
    vx, vy, vz = (aux[sel, AUX_VX].astype(np.float64), aux[sel, AUX_VY].astype(np.float64),
                  aux[sel, AUX_VZ].astype(np.float64))

    # ---- anchor branch, exactly as the slice builder / metrics compute it ----
    R_face, Z_face = load_front_face(args.geometry)
    a_eta, a_phi, mode = core_anchor(pt, p_phi, p_eta, qch, vx, vy, vz, R_face, Z_face, kind="helix")
    print(f"front face r={R_face:.0f} mm, |z|={Z_face:.0f} mm; branches "
          f"{ {BRANCH[c]: round(float((mode == c).mean()), 4) for c in (0,1,2,3)} }")

    # ---- shower core, defined EXACTLY as build_calo_slice.py does it (unweighted mean of the
    # particle-relative cell offsets, phi wrapped) so the spreads below are comparable with the
    # logged Phase 0b / Phase 1 numbers ----
    ch = d["calo_hits_flat"]
    ci, cown, cn = csr_gather(coff, sel)
    nc = np.maximum(cn, 1).astype(np.float64)
    core_eta = np.bincount(cown, weights=(ch[ci, CH_ETA] - p_eta[cown]).astype(np.float64),
                           minlength=len(sel)) / nc
    core_phi = np.bincount(cown, weights=wrap_pi(ch[ci, CH_PHI] - p_phi[cown]).astype(np.float64),
                           minlength=len(sel)) / nc
    del ch, ci, cown

    th = d["tracker_hits_flat"]

    # ---- Q1 coverage + the outermost real hit ----
    last, nhit = csr_last(toff, sel, th[:, TH_R])
    has = nhit > 0
    li = np.where(has, last, 0)
    h_r = th[li, TH_R].astype(np.float64); h_phi = th[li, TH_PHI].astype(np.float64)
    h_z = th[li, TH_Z].astype(np.float64)
    # eta of the outermost hit; the endpoint anchor in the same (eta, phi) coordinates as the core
    t_eta = np.arcsinh(np.divide(h_z, np.maximum(h_r, 1e-9)))
    t_phi = h_phi

    # ---- helix turning-point radius, for the Q2 comparison ----
    from genpu.helix import circle_params
    cx, cy, Rc = circle_params(pt, p_phi, qch, vx, vy)
    r_turn = np.hypot(cx, cy) + Rc                     # outermost radius the vacuum circle reaches

    out = {"tag": args.tag, "pdg_class": args.pdg_class, "shard": args.shard,
           "n_showers": int(len(sel)), "front_face": [float(R_face), float(Z_face)],
           "branch_frac": {BRANCH[c]: float((mode == c).mean()) for c in (0, 1, 2, 3)},
           "branches": {}}

    print(f"\nQ1 COVERAGE — tracker hits per particle, by anchor branch")
    print(f"{'branch':>8} {'showers':>9} {'no hits':>8} {'>=1':>6} {'>=3':>6} {'>=5':>6} "
          f"{'n_hits p10/25/50/75/90':>26} {'med pT':>7}")
    for c in (0, 1, 2):
        m = mode == c
        if m.sum() < 100:
            continue
        nh = nhit[m]
        print(f"{BRANCH[c]:>8} {int(m.sum()):>9} {float((nh == 0).mean()):>8.3f} "
              f"{float((nh >= 1).mean()):>6.3f} {float((nh >= 3).mean()):>6.3f} "
              f"{float((nh >= 5).mean()):>6.3f} "
              f"{'/'.join(str(int(x)) for x in np.percentile(nh, [10,25,50,75,90])):>26} "
              f"{float(np.median(pt[m])):>7.3f}")
        out["branches"][BRANCH[c]] = {
            "showers": int(m.sum()), "frac_no_hits": float((nh == 0).mean()),
            "frac_ge1": float((nh >= 1).mean()), "frac_ge3": float((nh >= 3).mean()),
            "frac_ge5": float((nh >= 5).mean()),
            "nhit_pct": [int(x) for x in np.percentile(nh, [10, 25, 50, 75, 90])],
            "median_pt": float(np.median(pt[m]))}

    print(f"\nQ2 ANCHOR ERROR — vacuum-helix turning point vs the OUTERMOST REAL HIT "
          f"(curler branch; what the helix cannot know)")
    mt = (mode == 2) & has
    if mt.sum() > 100:
        dr = h_r[mt] - r_turn[mt]
        dphi = wrap_pi(t_phi[mt] - a_phi[mt])
        deta = t_eta[mt] - a_eta[mt]
        print(f"  n = {int(mt.sum())}")
        print(f"  {q(r_turn[mt], 'helix r_turn [mm]      p10/25/50/75/90')}")
        print(f"  {q(h_r[mt],   'real outermost r [mm]  p10/25/50/75/90')}")
        print(f"  {q(dr,        'delta r [mm]           p10/25/50/75/90', '{:+.1f}')}")
        print(f"  |d_phi| median {np.median(np.abs(dphi)):.4f} rad, "
              f"|d_eta| median {np.median(np.abs(deta)):.4f}")
        out["turning_anchor_error"] = {
            "n": int(mt.sum()), "r_turn_med": float(np.median(r_turn[mt])),
            "r_real_med": float(np.median(h_r[mt])), "dr_med": float(np.median(dr)),
            "abs_dphi_med": float(np.median(np.abs(dphi))),
            "abs_deta_med": float(np.median(np.abs(deta)))}

    print(f"\nQ3 DOES THE ENDPOINT BEAT THE HELIX? — spread of (core - anchor), per branch")
    print(f"  (helix = the current Phase 1 anchor; track = the outermost real tracker hit)")
    print(f"{'branch':>8} {'n':>8} {'sig_phi helix':>14} {'sig_phi track':>14} {'gain':>6} "
          f"{'sig_eta helix':>14} {'sig_eta track':>14} {'gain':>6}")
    for c in (0, 1, 2):
        m = (mode == c) & has
        if m.sum() < 200:
            continue
        # residual of the shower core about each anchor, in the particle-relative frame
        rp_h = wrap_pi(core_phi[m] - a_phi[m]); re_h = core_eta[m] - a_eta[m]
        rp_t = wrap_pi(core_phi[m] - wrap_pi(t_phi[m] - p_phi[m]))
        re_t = core_eta[m] - (t_eta[m] - p_eta[m])
        # robust scale (the anchored core is sharply peaked with outliers -- Phase 0's caveat)
        def rs(a):
            p25, p75 = np.percentile(a, [25, 75])
            return float((p75 - p25) / 1.349)
        sph, spt = rs(rp_h), rs(rp_t); seh, set_ = rs(re_h), rs(re_t)
        print(f"{BRANCH[c]:>8} {int(m.sum()):>8} {sph:>14.4f} {spt:>14.4f} "
              f"{sph/max(spt,1e-9):>6.2f} {seh:>14.4f} {set_:>14.4f} {seh/max(set_,1e-9):>6.2f}")
        out["branches"].setdefault(BRANCH[c], {}).update({
            "sig_phi_helix": sph, "sig_phi_track": spt, "phi_gain_track_over_helix": sph / max(spt, 1e-9),
            "sig_eta_helix": seh, "sig_eta_track": set_, "eta_gain_track_over_helix": seh / max(set_, 1e-9)})

    o = Path(args.outdir); o.mkdir(parents=True, exist_ok=True)
    (o / f"curler_track_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", o / f"curler_track_{args.tag}.json")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        bins = np.arange(0, 26) - 0.5
        for c, col in ((0, C1), (1, C2), (2, C3)):
            m = mode == c
            if m.sum() < 100:
                continue
            ax[0].hist(np.clip(nhit[m], 0, 25), bins=bins, histtype="step", lw=2, color=col,
                       density=True, label=BRANCH[c])
        ax[0].set_xlabel("tracker hits on the particle"); ax[0].set_ylabel("density")
        ax[0].set_title(f"{args.tag}: tracker coverage by anchor branch")
        ax[0].legend(frameon=False)
        if mt.sum() > 100:
            ax[1].hist(np.clip(r_turn[mt], 0, 1600), bins=60, histtype="step", lw=2, color=C1,
                       density=True, label="helix turning point")
            ax[1].hist(np.clip(h_r[mt], 0, 1600), bins=60, histtype="step", lw=2, color=C2,
                       density=True, label="outermost real hit")
            ax[1].axvline(R_face, color="#888888", lw=1, ls="--")
            ax[1].annotate("calo face", (R_face, 0.98), xycoords=("data", "axes fraction"),
                           ha="right", va="top", rotation=90, color="#666666", fontsize=9)
        ax[1].set_xlabel("radius [mm]"); ax[1].set_ylabel("density")
        ax[1].set_title("curlers: where the helix says vs where the track got")
        ax[1].legend(frameon=False)
        for a_ in ax:
            a_.grid(alpha=0.25, lw=0.6); a_.set_axisbelow(True)
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
        fig.tight_layout()
        p = o / f"curler_track_{args.tag}.png"
        fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    main()
