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


def build(shards, preproc_dir, pdg_classes, min_hits, energy_mode):
    # cont features follow genpu.conditioning.CONT_FEATURES:
    #   log_pt, eta, log_E, charge, mass, vr, vz
    cont_list, pdg_list, glob_list, pts_list, lengths = [], [], [], [], []
    for sh in shards:
        f = Path(preproc_dir) / f"shard_{sh:04d}_stage2.npz"
        if not f.exists():
            print(f"  missing {f}, skip"); continue
        d = np.load(f)
        pf, aux = d["particle_features"], d["particle_aux"]
        ch, off = d["calo_hits_flat"], d["calo_offsets"]
        sel = np.where(np.isin(pf[:, PF_PDG], pdg_classes))[0]
        n_keep = 0
        for i in sel:
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
            d_eta = (hits[:, CH_ETA] - p_eta).astype(np.float32)
            d_phi = wrap_pi(hits[:, CH_PHI] - pf[i, PF_PHI]).astype(np.float32)
            if energy_mode == "abs":
                e_coord = np.log(np.clip(e, 1e-12, None)).astype(np.float32)
            else:  # "frac"
                e_coord = np.log(np.clip(e / tot, 1e-12, None)).astype(np.float32)
            pts = np.stack([d_eta, d_phi, e_coord], axis=1)
            log_E_part = np.log(max(float(aux[i, AUX_ENERGY]), 1e-6))
            vr = float(np.hypot(aux[i, AUX_VX], aux[i, AUX_VY]))
            # shared conditioning contract (phi excluded — response is phi-invariant)
            cont_list.append([pf[i, PF_LOGPT], p_eta, log_E_part, pf[i, PF_CHARGE],
                              pf[i, PF_MASS], vr, aux[i, AUX_VZ]])
            pdg_list.append(pf[i, PF_PDG])
            glob_list.append([np.log(tot), np.log(n)])
            pts_list.append(pts)
            lengths.append(n)
            n_keep += 1
        print(f"  shard {sh}: {len(sel)} pdg={pdg_classes} particles -> {n_keep} showers (>= {min_hits} hits)")
    cont = np.asarray(cont_list, np.float32)
    pdg = np.asarray(pdg_list, np.float32)
    glob = np.asarray(glob_list, np.float32)
    pts = np.concatenate(pts_list).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    return cont, pdg, glob, pts, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon_absE.npz")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    # PDG classes (see preprocessing.py): 2=photon; 3,4=pi+-; 5,6=K+-; 7,8=p; 9,10=n
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[2])
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--energy_mode", choices=["abs", "frac"], default="abs")
    args = ap.parse_args()

    cont, pdg, glob, pts, offsets = build(args.shards, args.preproc_dir, args.pdg_class, args.min_hits, args.energy_mode)
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
                        **norm)

    npt = np.diff(offsets)
    print(f"\nwrote {out}")
    print(f"  showers: {S}   points: {pts.shape[0]}   pts/shower mean={npt.mean():.1f} median={np.median(npt):.0f} max={npt.max()}")
    print(f"  total_logE: mean={glob[:,0].mean():.2f} std={glob[:,0].std():.2f}  (E GeV median={np.exp(np.median(glob[:,0])):.3e})")
    print(f"  d_eta std={pts[:,0].std():.4f}  d_phi std={pts[:,1].std():.4f}  log_efrac mean={pts[:,2].mean():.2f} std={pts[:,2].std():.2f}")
    # print a compact summary json next to it
    summ = {"showers": int(S), "points": int(pts.shape[0]),
            "pts_per_shower_mean": float(npt.mean()), "pts_per_shower_max": int(npt.max()),
            "pdg_class": args.pdg_class, "shards": args.shards}
    Path(str(out) + ".summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
