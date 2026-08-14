"""Build a per-particle calo-shower slice for the M2 flow spike.

Reads the existing stage2 preprocessed npz (particle_features / calo_hits_flat /
calo_offsets), selects one PDG class (default: photon=2) with >=1 calo deposit,
and writes per-shower point clouds in a shower-centred, floor-aware frame:

  per point:  (d_eta, d_phi, energy_coord)
     d_eta   = cell_eta - particle_eta
     d_phi   = wrap(cell_phi - particle_phi)   in (-pi, pi]
     energy_coord depends on --energy_mode:
       "abs"  (default): log(e_cell) absolute [GeV] — has the physical floor at
               log(FLOOR); clamp to that floor at generation to recover the point mass
       "frac": log(e_cell / sum_cells e)  (old behaviour; smears below the floor)
  per shower (global): total_logE = log(sum e),  n_points

  conditioning:  [log_pt, eta, log_E_particle, vz, charge]

Output npz:
  cond          (S, 4)         float32   per-shower conditioning
  glob          (S, 2)         float32   [total_logE, log_n_points]
  points_flat   (P, 3)         float32   [d_eta, d_phi, log_efrac]
  offsets       (S+1,)         int32     shower s owns points [off[s]:off[s+1]]
  norm          dict-ish arrays: means/stds for cond, glob, points (for standardisation)
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

# feature column indices (see src/genpu/preprocessing.py)
PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)


def wrap_pi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2 * np.pi) - np.pi


FLOOR_GEV = 5e-5  # zero-suppression floor on calo cell energy (see M0 data-QA)
WIDTH_FLOOR = 1e-5  # smallest per-shower width kept before log (n==1 showers have width 0)


def build(shards, preproc_dir, pdg_classes, min_hits, energy_mode, max_per_class=0, seed=0,
          width_norm=False, core_anchor="none", geometry=None):
    # cont features follow genpu.conditioning.CONT_FEATURES:
    #   log_pt, eta, log_E, charge, mass, vr, vz
    cont_list, pdg_list, glob_list, pts_list, lengths = [], [], [], [], []
    anchor_list, anchor_ok_list = [], []
    rng = np.random.default_rng(seed)
    kept_per_class: dict[int, int] = {}
    face = None
    if core_anchor != "none":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from genpu.calo_geom import load_front_face, core_anchor as anchor_of
        face = load_front_face(geometry)
        print(f"core anchor: truth {core_anchor} to the calo front face (barrel r={face[0]:.0f} mm, "
              f"endcap |z|={face[1]:.0f} mm)")
    for sh in shards:
        f = Path(preproc_dir) / f"shard_{sh:04d}_stage2.npz"
        if not f.exists():
            print(f"  missing {f}, skip"); continue
        d = np.load(f)
        pf, aux = d["particle_features"], d["particle_aux"]
        ch, off = d["calo_hits_flat"], d["calo_offsets"]
        ncell = np.diff(off)
        cls = pf[:, PF_PDG]
        # Per-class quota. On the multi-species mixture e- + e+ are 57% of all calo cells, so an
        # uncapped slice both bloats (a per-point copy of the conditioning is held on the GPU) and
        # lets the majority classes set the shared heads. Capping the big classes flattens the
        # mixture without up-weighting a 400-shower class into overfitting.
        quota = max_per_class // max(len(shards), 1) if max_per_class else 0
        picks = []
        for c in np.unique(cls[np.isin(cls, pdg_classes)]):
            cand = np.where((cls == c) & (ncell >= min_hits))[0]
            if quota and len(cand) > quota:
                cand = rng.choice(cand, quota, replace=False)
            picks.append(cand)
        sel = np.sort(np.concatenate(picks)) if picks else np.array([], int)
        # CORE ANCHOR (Phase 1): truth-helix extrapolation to the calo front face, in the particle
        # frame. Vectorised over the whole selection — the per-shower loop only indexes it.
        if face is not None and len(sel):
            anc_eta, anc_phi, anc_mode = anchor_of(
                np.exp(pf[sel, PF_LOGPT]), pf[sel, PF_PHI], pf[sel, PF_ETA], pf[sel, PF_CHARGE],
                aux[sel, AUX_VX], aux[sel, AUX_VY], aux[sel, AUX_VZ], face[0], face[1],
                kind=core_anchor)
        else:
            anc_eta = anc_phi = np.zeros(len(sel)); anc_mode = np.zeros(len(sel), np.int8)
        n_keep = 0
        for si, i in enumerate(sel):
            a, b = off[i], off[i + 1]
            n = b - a
            if n < min_hits:
                continue
            hits = ch[a:b]
            e = np.exp(hits[:, CH_LOGE]).astype(np.float64)       # GeV contrib
            tot = e.sum()
            if tot <= 0:
                continue
            p_eta = pf[i, PF_ETA]
            # deltas are taken from the ANCHOR when one is in use (anchor == 0 reproduces the
            # particle-direction frame exactly, including for invalid extrapolations), so `core`
            # below is the helix RESIDUAL and the point cloud `q` is unchanged either way.
            d_eta = (hits[:, CH_ETA] - (p_eta + anc_eta[si])).astype(np.float32)
            d_phi = wrap_pi(hits[:, CH_PHI] - (pf[i, PF_PHI] + anc_phi[si])).astype(np.float32)
            if energy_mode == "abs":
                e_coord = np.log(np.clip(e, 1e-12, None)).astype(np.float32)
            else:  # "frac"
                e_coord = np.log(np.clip(e / tot, 1e-12, None)).astype(np.float32)
            # per-shower CORE (geometric centroid of the points) goes in the global;
            # points are stored as tight DELTAS from the core so the flow learns
            # per-shower compactness (fixes the i.i.d.-points width defect).
            core_eta = float(d_eta.mean()); core_phi = float(d_phi.mean())
            q_eta = d_eta - core_eta; q_phi = d_phi - core_phi
            # per-shower WIDTH: the RMS size of the core-relative cloud. Measured 2026-08-13, real
            # widths span ~50x from q10 to q90 while generated ones sat in a band 0.79x as wide
            # (i.i.d. cells given (cond, global) can't produce per-shower scale variation, and the
            # particle features don't predict compactness). With width_norm the points are stored in
            # UNITS OF THE SHOWER'S OWN WIDTH and the width rides in the global, so the point flow
            # only models SHAPE and the width distribution is reproduced by the mixture — exactly
            # how the core already removed per-shower LOCATION variance from the flow.
            width = float(np.sqrt(np.mean(q_eta ** 2 + q_phi ** 2)))
            if width_norm:
                # n==1 (and degenerate identical-cell) showers have no size: the floor keeps the
                # log finite and their normalised points are exactly 0, so they regenerate at the core
                w = max(width, WIDTH_FLOOR)
                pts = np.stack([q_eta / w, q_phi / w, e_coord], axis=1)
            else:
                pts = np.stack([q_eta, q_phi, e_coord], axis=1)
            log_E_part = np.log(max(float(aux[i, AUX_ENERGY]), 1e-6))
            vr = float(np.hypot(aux[i, AUX_VX], aux[i, AUX_VY]))
            # shared conditioning contract (phi excluded — response is phi-invariant)
            cont_list.append([pf[i, PF_LOGPT], p_eta, log_E_part, pf[i, PF_CHARGE],
                              pf[i, PF_MASS], vr, aux[i, AUX_VZ]])
            pdg_list.append(pf[i, PF_PDG])
            # glob layout is APPEND-ONLY (downstream code slices dims 2:4 for the core):
            #   [total_logE, log_n, core_eta, core_phi] (+ log_width when width_norm)
            gl = [np.log(tot), np.log(n), core_eta, core_phi]
            if width_norm:
                gl.append(np.log(max(width, WIDTH_FLOOR)))
            glob_list.append(gl)
            pts_list.append(pts)
            anchor_list.append([anc_eta[si], anc_phi[si]]); anchor_ok_list.append(anc_mode[si])
            lengths.append(n)
            n_keep += 1
            kept_per_class[int(pf[i, PF_PDG])] = kept_per_class.get(int(pf[i, PF_PDG]), 0) + 1
        print(f"  shard {sh}: {len(sel)} selected particles -> {n_keep} showers (>= {min_hits} hits)"
              + (f", quota {quota}/class" if quota else ""), flush=True)
    cont = np.asarray(cont_list, np.float32)
    pdg = np.asarray(pdg_list, np.float32)
    glob = np.asarray(glob_list, np.float32)
    pts = np.concatenate(pts_list).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    anchor = np.asarray(anchor_list, np.float32)
    anchor_mode = np.asarray(anchor_ok_list, np.int8)
    return cont, pdg, glob, pts, offsets, kept_per_class, anchor, anchor_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon_absE.npz")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    # PDG classes (see preprocessing.py): 2=photon; 3,4=pi+-; 5,6=K+-; 7,8=p; 9,10=n
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[2])
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--energy_mode", choices=["abs", "frac"], default="abs")
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="0=no cap; else keep at most N showers per PDG class (multi-species slices)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width_norm", action="store_true",
                    help="store points in units of each shower's own RMS width and put log_width in "
                         "the global (glob dim 4). Fixes the per-shower width distribution the "
                         "i.i.d. point flow cannot produce.")
    ap.add_argument("--core_anchor", choices=["none", "helix", "line", "auto"], default="none",
                    help="'helix': store the shower core as a RESIDUAL from the truth-helix "
                         "extrapolation to the calo front face (Phase 1). Truth conditioning only, "
                         "so generation reconstructs the anchor exactly. Measured phi tightening: "
                         "18.8x/15.2x pion on the face branches, 2.5x on curlers. 'line' forces "
                         "the STRAIGHT-LINE path for every particle, charge included — the "
                         "bremsstrahlung hypothesis for e± (radiated photons fly straight from "
                         "the radiation point, and the helix anchor measurably gets worse the "
                         "stiffer the e± track: gain 4.52x -> 0.43x across pT).")
    ap.add_argument("--geometry", default=None,
                    help="calo_geometry.json (default: repo root); only used with --core_anchor helix")
    args = ap.parse_args()

    cont, pdg, glob, pts, offsets, per_class, anchor, anchor_mode = build(
        args.shards, args.preproc_dir, args.pdg_class, args.min_hits, args.energy_mode,
        max_per_class=args.max_per_class, seed=args.seed, width_norm=args.width_norm,
        core_anchor=args.core_anchor, geometry=args.geometry)
    S = cont.shape[0]

    # standardisation stats (fit on this slice; saved for train/eval).
    # charge/mass have ~zero variance in single-species slices -> +1e-6 guards it.
    norm = {
        "cont_mean": cont.mean(0), "cont_std": cont.std(0) + 1e-6,
        "glob_mean": glob.mean(0), "glob_std": glob.std(0) + 1e-6,
        "pts_mean": pts.mean(0), "pts_std": pts.std(0) + 1e-6,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, pdg=pdg, glob=glob, points_flat=pts, offsets=offsets,
                        energy_mode=np.array(args.energy_mode), log_floor=np.array(np.log(FLOOR_GEV), np.float32),
                        width_normalized=np.array(1 if args.width_norm else 0, np.int32),
                        # the anchor is recomputed from truth at generation, never read back from
                        # here — it is stored so diagnostics can reconstruct the PHYSICAL core
                        # (residual + anchor) and compare against the pre-anchor numbers.
                        core_anchor=np.array(args.core_anchor), anchor=anchor,
                        anchor_mode=anchor_mode,   # 0 barrel, 1 endcap, 2 turning point, 3 none
                        **norm)

    npt = np.diff(offsets)
    print(f"\nwrote {out}")
    print(f"  showers: {S}   points: {pts.shape[0]}   pts/shower mean={npt.mean():.1f} median={np.median(npt):.0f} max={npt.max()}")
    print(f"  total_logE: mean={glob[:,0].mean():.2f} std={glob[:,0].std():.2f}  (E GeV median={np.exp(np.median(glob[:,0])):.3e})")
    print(f"  d_eta std={pts[:,0].std():.4f}  d_phi std={pts[:,1].std():.4f}  log_efrac mean={pts[:,2].mean():.2f} std={pts[:,2].std():.2f}")
    if args.core_anchor != "none":
        # what the anchor bought, in the slice's own coordinates: the stored core is the residual,
        # residual + anchor is the physical (particle-frame) core the un-anchored slice would hold.
        res_e, res_p = glob[:, 2], glob[:, 3]
        phys_e, phys_p = res_e + anchor[:, 0], wrap_pi(res_p + anchor[:, 1])
        def rsig(a):
            q1, q3 = np.percentile(a, [25, 75]); return (q3 - q1) / 1.349
        print(f"  core_anchor={args.core_anchor} branches: barrel {(anchor_mode==0).mean():.4f}  "
              f"endcap {(anchor_mode==1).mean():.4f}  turning point {(anchor_mode==2).mean():.4f}  "
              f"none {(anchor_mode==3).mean():.4f} (anchor=0, particle frame)")
        print(f"    |core| median  particle-frame {np.median(np.hypot(phys_e, phys_p)):.4f} "
              f"-> anchored {np.median(np.hypot(res_e, res_p)):.4f}")
        print(f"    robust sigma   phi {rsig(phys_p):.4f} -> {rsig(res_p):.4f} "
              f"({rsig(phys_p)/max(rsig(res_p),1e-9):.2f}x tighter) | "
              f"eta {rsig(phys_e):.4f} -> {rsig(res_e):.4f} "
              f"({rsig(phys_e)/max(rsig(res_e),1e-9):.2f}x)")
        for c, lab in [(0, "barrel"), (1, "endcap"), (2, "turning")]:
            s = anchor_mode == c
            if s.sum() < 200:
                continue
            print(f"      {lab:>8}: n={int(s.sum()):>8}  sigma(phi) {rsig(phys_p[s]):.4f} -> "
                  f"{rsig(res_p[s]):.4f} ({rsig(phys_p[s])/max(rsig(res_p[s]),1e-9):.2f}x)")
    if args.width_norm:
        w = np.exp(glob[:, 4])
        print(f"  width_norm ON: log_width mean={glob[:,4].mean():.2f} std={glob[:,4].std():.2f} | "
              f"width q10={np.percentile(w,10):.4f} q50={np.percentile(w,50):.4f} q90={np.percentile(w,90):.4f} "
              f"(spread q90/q10={np.percentile(w,90)/max(np.percentile(w,10),1e-9):.1f}x)")
        rms = np.sqrt(np.bincount(np.repeat(np.arange(S), npt), weights=pts[:,0]**2+pts[:,1]**2,
                                  minlength=S) / np.maximum(npt, 1))
        multi = npt > 1
        print(f"  normalised-cloud RMS (should be 1 for n>1): mean={rms[multi].mean():.4f} "
              f"std={rms[multi].std():.4f}   n==1 showers: {int((npt==1).sum())} ({100*(npt==1).mean():.1f}%)")
    # print a compact summary json next to it
    if len(per_class) > 1:
        print("  showers per PDG class: " + ", ".join(
            f"{c}:{n}" for c, n in sorted(per_class.items(), key=lambda kv: -kv[1])))
    summ = {"showers": int(S), "points": int(pts.shape[0]),
            "pts_per_shower_mean": float(npt.mean()), "pts_per_shower_max": int(npt.max()),
            "pdg_class": args.pdg_class, "shards": args.shards,
            "max_per_class": args.max_per_class, "showers_per_class": per_class,
            "core_anchor": args.core_anchor,
            "anchor_branch_frac": ({lab: float((anchor_mode == c).mean()) for c, lab in
                                    [(0, "barrel"), (1, "endcap"), (2, "turning"), (3, "none")]}
                                   if len(anchor_mode) else {})}
    Path(str(out) + ".summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
