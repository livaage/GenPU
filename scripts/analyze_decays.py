"""M0 decay characterization: what must the decay-injection step reproduce?

Reads the pu0 particle truth (parent_id, primary, production vertex, kinematics)
and characterises in-flight decays:
  - fraction of particles that are secondaries (have a parent)
  - production-radius distribution of secondaries (where decays happen)
  - which PARENT species produce displaced daughters, and the daughter pdg/count
This is the target the analytic decay-point sampler (ctau/betagamma) must match,
per the plan's M0 sub-step, BEFORE response heads train on daughters.
"""
from __future__ import annotations
import argparse, glob
from collections import Counter
import numpy as np
import pyarrow as pa

PDG_NAME = {211: "pi+", -211: "pi-", 321: "K+", -321: "K-", 310: "K0S", 130: "K0L",
            2212: "p", 2112: "n", 11: "e-", -11: "e+", 22: "gamma", 13: "mu-", -13: "mu+",
            3122: "Lambda", -3122: "aLambda", 3222: "Sigma+", 3112: "Sigma-", 111: "pi0"}


def load(sub, shard):
    base = "/scratch/gpfs/IOJALVO/lv7805/genpu_cache/CERN___collider_ml-release-1"
    f = sorted(glob.glob(f"{base}/pileup_only_pu0_{sub}/*/*/*.arrow"))[shard]
    try:
        return pa.ipc.open_file(pa.memory_map(f)).read_all()
    except pa.lib.ArrowInvalid:
        return pa.ipc.open_stream(pa.memory_map(f)).read_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_events", type=int, default=2000)
    args = ap.parse_args()
    t = load("particles", args.shard)
    N = min(args.n_events, t.num_rows)

    pid = t.column("particle_id").to_pylist()
    parent = t.column("parent_id").to_pylist()
    pdg = t.column("pdg_id").to_pylist()
    prim = t.column("primary").to_pylist()
    vx = t.column("vx").to_pylist(); vy = t.column("vy").to_pylist(); vz = t.column("vz").to_pylist()
    px = t.column("px").to_pylist(); py = t.column("py").to_pylist(); pz = t.column("pz").to_pylist()

    n_tot = 0; n_sec = 0
    sec_r = []                       # secondary production radius (mm)
    parent_species = Counter()       # pdg of parents that have displaced daughters
    daughter_species = Counter()
    ndau = Counter()                 # daughters per decaying parent (pdg -> list)
    parent_of_displaced = Counter()  # parent pdg for daughters at r>20mm

    for ev in range(N):
        ids = np.asarray(pid[ev], np.int64)
        par = np.asarray(parent[ev], np.int64)
        pg = np.asarray(pdg[ev], np.int64)
        pr = np.asarray(prim[ev])
        r = np.hypot(np.asarray(vx[ev]), np.asarray(vy[ev]))
        n_tot += len(ids)
        id_to_idx = {int(i): k for k, i in enumerate(ids)}
        # daughters per parent
        dau_by_parent = {}
        for k in range(len(ids)):
            is_sec = (not bool(pr[k])) or (int(par[k]) in id_to_idx)
            if is_sec:
                n_sec += 1
                sec_r.append(float(r[k]))
                p_idx = id_to_idx.get(int(par[k]))
                if p_idx is not None:
                    dau_by_parent.setdefault(p_idx, []).append(k)
        for p_idx, dks in dau_by_parent.items():
            ppdg = int(pg[p_idx])
            # displaced = daughters produced away from beamline (in-flight decay/interaction)
            if any(r[dk] > 20 for dk in dks):
                parent_species[ppdg] += 1
                parent_of_displaced[ppdg] += 1
                for dk in dks:
                    daughter_species[int(pg[dk])] += 1
                ndau[ppdg] += len(dks)

    sec_r = np.array(sec_r)
    print(f"events={N}  particles={n_tot}  secondaries={n_sec} ({n_sec/n_tot:.1%})")
    print(f"\nsecondary production radius (mm): median={np.median(sec_r):.1f} "
          f"p90={np.percentile(sec_r,90):.0f} max={sec_r.max():.0f}")
    for lo, hi in [(0,1),(1,20),(20,200),(200,600),(600,1100),(1100,1e9)]:
        f = ((sec_r>=lo)&(sec_r<hi)).mean()
        print(f"  r in [{lo:5.0f},{hi:5.0f}) mm : {f:6.1%}")

    def show(counter, title, topn=12):
        print(f"\n{title}")
        for k, c in counter.most_common(topn):
            print(f"  {PDG_NAME.get(k,str(k)):8s} ({k:5d}) : {c}")
    show(parent_of_displaced, "PARENT species with displaced (r>20mm) daughters (in-flight decays/interactions):")
    show(daughter_species, "DAUGHTER species from those displaced decays:")
    print("\navg daughters per displaced parent:")
    for k, c in parent_of_displaced.most_common(8):
        print(f"  {PDG_NAME.get(k,str(k)):8s} : {ndau[k]/c:.2f}")


if __name__ == "__main__":
    main()
