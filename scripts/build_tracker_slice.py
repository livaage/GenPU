"""Build a per-particle tracker-hit slice for the AR tracker head.

Reads the existing stage2 preprocessed npz (particle_features / particle_aux /
tracker_hits_flat / tracker_offsets), selects one PDG class set (default: charged
pions [3,4]) with >=1 tracker hit, and writes per-particle hit sequences in the
per-layer-standardized residual frame the AR model consumes:

  per hit:  (layer_class, r_resid, phi_resid, z_resid, time_resid)
     layer_class  = detector layer index 0..47 (see detector_geometry.LAYER_LIST)
     *_resid      = (physical - LAYER_MEANS[layer_class]) / LAYER_STDS[layer_class]
                    so every residual is ~N(0,1) regardless of layer type
     hits are SORTED within the particle by physical r ascending (inner->outer),
     matching the AR generation order.

  conditioning:  the 7 shared CONT_FEATURES (see genpu.conditioning):
     [log_pt, eta, log_E(=log aux energy), charge, mass, vr(=hypot(vx,vy)), vz]
     phi excluded — the response is phi-invariant by symmetry.

Note on data layout: the preprocessed tracker_hits_flat is (M,5) with column 0
ALREADY the layer_class (0..47) and columns 1..4 the RAW physical (r,phi,z,time).
(An older 6-col [r,phi,z,time,volume_id,layer_id] layout is handled as a fallback.)

Output npz:
  cont          (S, 7)         float32   per-particle conditioning
  pdg           (S,)           float32   per-particle PDG class
  hits          (P, 5)         float32   [layer_class, r_resid, phi_resid, z_resid, time_resid]
  offsets       (S+1,)         int32     particle s owns hits [off[s]:off[s+1]]
  cont_mean/cont_std           float32   standardisation stats for cont
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS, hits_to_layer_class

# feature column indices (see src/genpu/preprocessing.py)
PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)


def _split_hits(hits: np.ndarray):
    """Return (layer_class int, physical (n,4) [r,phi,z,time]) from a raw hit block.

    Real data: (n,5) = [layer_class, r, phi, z, time].
    Fallback:  (n,6) = [r, phi, z, time, volume_id, layer_id].
    """
    if hits.shape[1] == 5:
        lc = np.round(hits[:, 0]).astype(np.int32).clip(0, N_LAYERS - 1)
        phys = hits[:, 1:5].astype(np.float32)
    else:  # (n,6) fallback per the original spec
        lc = hits_to_layer_class(hits[:, 4], hits[:, 5]).astype(np.int32)
        phys = hits[:, 0:4].astype(np.float32)
    return lc, phys


def build(shards, preproc_dir, pdg_classes, min_hits, population="all"):
    # cont features follow genpu.conditioning.CONT_FEATURES:
    #   log_pt, eta, log_E, charge, mass, vr, vz
    cont_list, pdg_list, hits_list, lengths = [], [], [], []
    occ = np.zeros(N_LAYERS, dtype=np.int64)  # layer occupancy sanity
    for sh in shards:
        f = Path(preproc_dir) / f"shard_{sh:04d}_stage2.npz"
        if not f.exists():
            print(f"  missing {f}, skip"); continue
        d = np.load(f)
        pf, aux = d["particle_features"], d["particle_aux"]
        th, off = d["tracker_hits_flat"], d["tracker_offsets"]
        mask = np.isin(pf[:, PF_PDG], pdg_classes)
        if population == "primary":
            mask &= aux[:, AUX_PRIMARY] > 0.5           # truth primary flag
        elif population == "secondary":
            mask &= aux[:, AUX_PRIMARY] <= 0.5
        sel = np.where(mask)[0]
        n_keep = 0
        for i in sel:
            a, b = off[i], off[i + 1]
            n = b - a
            if n < min_hits:
                continue
            lc, phys = _split_hits(th[a:b])
            r_phys = phys[:, 0]
            # sort inner->outer by physical r (matches AR generation order)
            order = np.argsort(r_phys, kind="stable")
            lc, phys = lc[order], phys[order]
            # per-layer-standardized residuals (r, phi, z, time)
            resid = (phys - LAYER_MEANS[lc]) / LAYER_STDS[lc]
            hits = np.concatenate([lc[:, None].astype(np.float32), resid], axis=1)
            occ += np.bincount(lc, minlength=N_LAYERS)
            log_E_part = np.log(max(float(aux[i, AUX_ENERGY]), 1e-6))
            vr = float(np.hypot(aux[i, AUX_VX], aux[i, AUX_VY]))
            # shared conditioning contract (phi excluded — response is phi-invariant)
            cont_list.append([pf[i, PF_LOGPT], pf[i, PF_ETA], log_E_part, pf[i, PF_CHARGE],
                              pf[i, PF_MASS], vr, aux[i, AUX_VZ]])
            pdg_list.append(pf[i, PF_PDG])
            hits_list.append(hits)
            lengths.append(int(n))
            n_keep += 1
        print(f"  shard {sh}: {len(sel)} pdg={pdg_classes} particles -> {n_keep} particles (>= {min_hits} hits)")
    cont = np.asarray(cont_list, np.float32)
    pdg = np.asarray(pdg_list, np.float32)
    hits = np.concatenate(hits_list).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    return cont, pdg, hits, offsets, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/pion.npz")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    # PDG classes (see preprocessing.py): 2=photon; 3,4=pi+-; 5,6=K+-; 7,8=p; 9,10=n
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--population", choices=["all", "primary", "secondary"], default="all")
    args = ap.parse_args()

    cont, pdg, hits, offsets, occ = build(args.shards, args.preproc_dir, args.pdg_class, args.min_hits, args.population)
    S = cont.shape[0]

    # standardisation stats (fit on this slice; saved for train/eval).
    # charge/mass have ~zero variance in single-species slices -> +1e-6 guards it.
    norm = {
        "cont_mean": cont.mean(0), "cont_std": cont.std(0) + 1e-6,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, pdg=pdg, hits=hits, offsets=offsets, **norm)

    nph = np.diff(offsets)
    print(f"\nwrote {out}")
    print(f"  particles: {S}   hits: {hits.shape[0]}   hits/particle mean={nph.mean():.1f} median={np.median(nph):.0f} max={nph.max()}")
    print(f"  r_resid std={hits[:,1].std():.3f}  phi_resid std={hits[:,2].std():.3f}  "
          f"z_resid std={hits[:,3].std():.3f}  time_resid std={hits[:,4].std():.3f}")
    nz = np.nonzero(occ)[0]
    print(f"  layer occupancy: {len(nz)}/{N_LAYERS} layers hit  "
          f"(busiest classes: {list(np.argsort(occ)[::-1][:5])})")
    # print a compact summary json next to it
    summ = {"particles": int(S), "hits": int(hits.shape[0]),
            "hits_per_particle_mean": float(nph.mean()), "hits_per_particle_max": int(nph.max()),
            "pdg_class": args.pdg_class, "shards": args.shards,
            "layers_occupied": int(len(nz))}
    Path(str(out) + ".summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
