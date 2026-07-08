"""How much of the detector response comes from SECONDARIES (displaced particles)?

Determines how severe the material-effect / secondary-injection problem is per
detector. Uses the preprocessed per-particle association: a particle is a
'secondary' if produced at vr>20mm (or primary flag false). Reports the fraction
of tracker hits, calo hits, and calo ENERGY attributable to secondaries, and the
same split by production-radius band.
"""
from __future__ import annotations
import argparse
import numpy as np

PF_PDG = 3
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_LOGE = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed/shard_0000_stage2.npz")
    args = ap.parse_args()
    d = np.load(args.npz)
    aux = d["particle_aux"]
    ntrk = np.diff(d["tracker_offsets"]).astype(np.int64)
    ncal = np.diff(d["calo_offsets"]).astype(np.int64)
    calo_e = np.exp(d["calo_hits_flat"][:, CH_LOGE]).astype(np.float64)   # per-hit contrib energy
    calo_off = d["calo_offsets"]

    vr = np.hypot(aux[:, AUX_VX], aux[:, AUX_VY])
    prim = aux[:, AUX_PRIMARY] > 0.5
    sec = (~prim) | (vr > 20.0)                    # secondary = displaced or not-primary

    # per-particle calo energy sum (bincount is robust to zero-hit particles)
    seg = np.repeat(np.arange(len(ncal)), ncal)          # particle index per calo hit
    cal_e_particle = np.bincount(seg, weights=calo_e, minlength=len(ncal))

    def frac(mask, w):
        return w[mask].sum() / max(w.sum(), 1e-30)

    print(f"particles={len(ntrk)}  primary={prim.mean():.1%}  secondary(displaced)={sec.mean():.1%}")
    print(f"\nfraction of TRACKER hits from secondaries: {frac(sec, ntrk):.1%}")
    print(f"fraction of CALO   hits from secondaries: {frac(sec, ncal):.1%}")
    print(f"fraction of CALO ENERGY from secondaries: {frac(sec, cal_e_particle):.1%}")

    print("\nby production-radius band (tracker-hits / calo-hits / calo-energy share):")
    bands = [(-1, 1), (1, 20), (20, 200), (200, 600), (600, 1100), (1100, 1e9)]
    for lo, hi in bands:
        m = (vr >= lo) & (vr < hi)
        print(f"  vr[{lo:6.0f},{hi:6.0f}) mm : "
              f"trk {frac(m, ntrk):5.1%}  cal {frac(m, ncal):5.1%}  calE {frac(m, cal_e_particle):5.1%}"
              f"   ({m.mean():5.1%} of particles)")

    # among tracker-hit-leaving particles, what fraction are secondaries?
    trkp = ntrk > 0
    print(f"\nof particles that leave >=1 TRACKER hit: { (sec & trkp).sum()/max(trkp.sum(),1):.1%} are secondaries")
    calp = ncal > 0
    print(f"of particles that leave >=1 CALO   hit: { (sec & calp).sum()/max(calp.sum(),1):.1%} are secondaries")


if __name__ == "__main__":
    main()
