"""Which particle species actually matter for the calorimeter?

Census over a preprocessed stage2 shard: per PDG class, how many particles deposit calo
hits, how many cells they own, and how much energy they carry. Answers "which species
must the calo head cover" (coverage is by ENERGY and by CELLS, not by particle count).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)

CLASS_NAME = {0: "e-", 1: "e+", 2: "gamma", 3: "pi+", 4: "pi-", 5: "K+", 6: "K-",
              7: "p", 8: "pbar", 9: "n", 10: "nbar", 11: "mu-", 12: "mu+",
              13: "K0L", 14: "K0S", 15: "pi0", 16: "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--shard", type=int, default=5)
    args = ap.parse_args()

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, ch, off = d["particle_features"], d["particle_aux"], d["calo_hits_flat"], d["calo_offsets"]
    cls = pf[:, PF_PDG].astype(int)
    ncell = np.diff(off).astype(np.int64)
    e_cell = np.exp(ch[:, CH_LOGE].astype(np.float64))
    # per-particle deposited energy
    csum = np.concatenate([[0.0], np.cumsum(e_cell)])
    e_part = csum[off[1:]] - csum[off[:-1]]

    tot_cells = ncell.sum(); tot_E = e_part.sum(); n_dep = (ncell > 0).sum()
    print(f"shard {args.shard}: {len(pf)} particles, {n_dep} with >=1 calo cell "
          f"({100*n_dep/len(pf):.1f}%), {tot_cells} cells, {tot_E:.1f} GeV deposited\n")
    hdr = f"{'class':>6} {'name':>6} {'particles':>10} {'w/dep':>9} {'dep%':>6} {'cells':>10} {'cell%':>6} {'E_GeV':>10} {'E%':>6} {'cells/sh':>9} {'medE_dep':>9}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for c in sorted(set(cls.tolist())):
        m = cls == c
        md = m & (ncell > 0)
        rows.append((e_part[m].sum(), c, m.sum(), md.sum(), ncell[m].sum(),
                     float(np.median(ncell[md])) if md.sum() else 0.0,
                     float(np.median(e_part[md])) if md.sum() else 0.0))
    for E, c, npart, ndep, ncl, medn, medE in sorted(rows, reverse=True):
        print(f"{c:>6} {CLASS_NAME.get(c,'?'):>6} {npart:>10} {ndep:>9} {100*ndep/max(npart,1):>5.1f}% "
              f"{ncl:>10} {100*ncl/tot_cells:>5.1f}% {E:>10.1f} {100*E/tot_E:>5.1f}% {medn:>9.0f} {medE:>9.2e}")

    # cumulative coverage by energy and by cells, in descending-energy order
    print("\ncumulative coverage (species added in descending energy order):")
    cumE = cumC = 0.0
    for E, c, npart, ndep, ncl, medn, medE in sorted(rows, reverse=True):
        cumE += 100 * E / tot_E; cumC += 100 * ncl / tot_cells
        print(f"  +{CLASS_NAME.get(c,'?'):>6}: energy {cumE:6.2f}%   cells {cumC:6.2f}%")

    # charged-vs-neutral hadron split matters for the response model
    for name, cc in [("EM (e±,γ,π0)", [0, 1, 2, 15]), ("charged had (π±,K±,p,pbar)", [3, 4, 5, 6, 7, 8]),
                     ("neutral had (n,nbar,K0L,K0S)", [9, 10, 13, 14]), ("muons", [11, 12])]:
        m = np.isin(cls, cc) & (ncell > 0)
        print(f"{name:>30}: {m.sum():>8} showers  {ncell[m].sum():>9} cells "
              f"({100*ncell[m].sum()/tot_cells:5.1f}%)  {e_part[m].sum():>8.1f} GeV ({100*e_part[m].sum()/tot_E:5.1f}%)")


if __name__ == "__main__":
    main()
