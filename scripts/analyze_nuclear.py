"""Cascade spike (hard case): characterize HADRON interactions in pu0 truth.

Charged pions / protons / neutrons producing DISPLACED daughters (in-flight
nuclear interactions + decays). Much messier than photon conversions: variable
multiplicity, diverse daughter species incl. nuclei. Measures what a generator
must reproduce, per parent species:
  - interaction fraction (has >=1 displaced daughter, r_dau > r_parent+20mm)
  - interaction radius, multiplicity distribution, daughter species mix
  - interaction fraction vs parent energy, energy balance sum(E_dau)/E_parent
"""
from __future__ import annotations
import argparse, glob
from collections import Counter
import numpy as np
import pyarrow as pa

PARENTS = {211: "pi+", -211: "pi-", 2212: "p", 2112: "n"}
NAME = {**PARENTS, 22: "gamma", 11: "e-", -11: "e+", 13: "mu-", -13: "mu+",
        321: "K+", -321: "K-", 111: "pi0", 130: "K0L", 310: "K0S"}


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
    ap.add_argument("--n_events", type=int, default=3000)
    ap.add_argument("--parent", type=int, default=211, help="parent pdg to characterize")
    args = ap.parse_args()
    t = load("particles", args.shard); N = min(args.n_events, t.num_rows)
    pid = t.column("particle_id").to_pylist(); parent = t.column("parent_id").to_pylist()
    pdg = t.column("pdg_id").to_pylist(); energy = t.column("energy").to_pylist()
    vx = t.column("vx").to_pylist(); vy = t.column("vy").to_pylist()

    n_par = 0; n_int = 0
    int_r = []; mult = []; efrac = []
    par_E = []; par_int = []
    dau_species = Counter()
    abspar = abs(args.parent)

    for ev in range(N):
        ids = np.asarray(pid[ev], np.int64); par = np.asarray(parent[ev], np.int64)
        pg = np.asarray(pdg[ev], np.int64); en = np.asarray(energy[ev], np.float64)
        r = np.hypot(np.asarray(vx[ev]), np.asarray(vy[ev]))
        idx = {int(i): k for k, i in enumerate(ids)}
        dau = {}
        for k in range(len(ids)):
            p = idx.get(int(par[k]))
            if p is not None:
                dau.setdefault(p, []).append(k)
        for k in range(len(ids)):
            if abs(pg[k]) != abspar:
                continue
            n_par += 1; par_E.append(en[k])
            disp = [dk for dk in dau.get(k, []) if r[dk] > r[k] + 20]
            par_int.append(1.0 if disp else 0.0)
            if disp:
                n_int += 1
                int_r.append(float(np.min([r[dk] for dk in disp])))
                mult.append(len(disp))
                efrac.append(float(sum(en[dk] for dk in disp) / max(en[k], 1e-9)))
                for dk in disp:
                    dau_species[int(pg[dk])] += 1

    int_r = np.array(int_r); mult = np.array(mult); efrac = np.array(efrac)
    par_E = np.array(par_E); par_int = np.array(par_int)
    print(f"parent={NAME.get(args.parent,args.parent)}  count={n_par}  interacted(displaced dau)={n_int} ({n_int/n_par:.1%})")
    if n_int == 0:
        return
    print(f"\ninteraction radius (mm): median={np.median(int_r):.0f} p10={np.percentile(int_r,10):.0f} p90={np.percentile(int_r,90):.0f}")
    print(f"multiplicity (# displaced daughters): mean={mult.mean():.2f} median={np.median(mult):.0f} max={mult.max()}")
    print("  mult distribution:", {int(m): int((mult==m).sum()) for m in range(1, min(mult.max()+1, 9))}, "...")
    print(f"energy balance sum(E_dau)/E_parent: median={np.median(efrac):.2f} p90={np.percentile(efrac,90):.2f}")
    print(f"\ninteraction fraction vs parent energy:")
    eb = np.quantile(par_E, np.linspace(0, 1, 6))
    for b in range(5):
        m = (par_E >= eb[b]) & (par_E < eb[b+1] + (1e-6 if b == 4 else 0))
        print(f"  E [{eb[b]:7.3f},{eb[b+1]:7.3f}) GeV : int_frac={par_int[m].mean():5.1%}  (n={m.sum()})")
    print("\ndaughter species mix:")
    tot = sum(dau_species.values())
    for k, c in dau_species.most_common(12):
        print(f"  {NAME.get(k, str(k)):8s} ({k:11d}) : {c/tot:5.1%}")


if __name__ == "__main__":
    main()
