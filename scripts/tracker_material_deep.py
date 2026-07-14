"""Understand tracker material interactions in depth. From the source (all species, true
positions, energy), test what drives how far a particle gets before interacting:
  1. PARTICLE TYPE   - survival (n_hits) per species. Muons penetrate; hadrons interact; e brem.
  2. INCIDENCE ANGLE - tracks crossing layers at a shallow angle see more material -> stop earlier.
  3. ENERGY          - higher energy -> penetrate further? (revisit, per species)
  4. DEPTH (energy loss proxy) - does per-layer survival DROP with depth as the particle slows?
Reports per-species n_hits + survival, and within pions the n_hits vs (incidence, log E, |eta|)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, explode_list_columns
from genpu.preprocessing import pdg_to_class

CLASS_NAME = {2: "photon", 3: "pi+", 4: "pi-", 5: "K+", 6: "K-", 7: "proton", 8: "antiproton",
              9: "neutron", 10: "antineutron", 11: "e-", 12: "e+", 13: "mu-", 14: "mu+", 0: "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max_particles", type=int, default=300000)
    args = ap.parse_args()
    rows = []            # (cls, n_hits, pt, logE, aeta, inc_deg, last_layer)
    for sh in args.shards:
        P = load_shard("particles", sh); T = load_shard("tracker_hits", sh)
        for e in range(min(P.num_rows, T.num_rows)):
            pe = explode_list_columns(P, e); te = explode_list_columns(T, e)
            cls = pdg_to_class(pe["pdg_id"].astype(np.int64))
            pt = np.hypot(pe["px"], pe["py"]); aeta = np.abs(np.arcsinh(pe["pz"] / (pt + 1e-9)))
            logE = np.log(np.clip(pe["energy"], 1e-6, None))
            tpid = te["particle_id"].astype(np.int64)
            for i in range(len(pe["particle_id"])):
                m = tpid == pe["particle_id"][i]
                nh = int(m.sum())
                if nh == 0:
                    continue
                # incidence at first hit: tangent(vertex->first true hit) vs radial
                tx, ty, tz = te["true_x"][m], te["true_y"][m], te["true_z"][m]
                r = np.hypot(tx, ty); o = np.argsort(r)
                fx, fy, fz = tx[o][0], ty[o][0], tz[o][0]
                dvec = np.array([fx - pe["vx"][i], fy - pe["vy"][i], fz - pe["vz"][i]])
                dvec /= (np.linalg.norm(dvec) + 1e-9)
                rhat = np.array([fx, fy, 0.0]); rhat /= (np.linalg.norm(rhat) + 1e-9)
                inc = np.degrees(np.arccos(np.clip(abs(np.dot(dvec, rhat)), 0, 1)))  # 0=radial, 90=tangential
                rows.append((int(cls[i]), nh, float(pt[i]), float(logE[i]), float(aeta[i]), float(inc)))
            if len(rows) >= args.max_particles:
                break
        if len(rows) >= args.max_particles:
            break
    a = np.array(rows, float)
    cls, nh, pt, logE, aeta, inc = a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]

    print("=" * 70); print(f"MATERIAL INTERACTION — DEEP ({len(a)} particles)"); print("=" * 70)
    print("1. PARTICLE TYPE (survival)")
    print(f"  {'species':10s} {'n':>8s} {'n_hits mean':>11s} {'median':>7s} {'frac>=5':>8s} {'frac>=10':>9s}")
    for c in sorted(set(cls)):
        s = cls == c
        if s.sum() < 200:
            continue
        print(f"  {CLASS_NAME.get(c,c):10s} {s.sum():8d} {nh[s].mean():11.2f} {np.median(nh[s]):7.0f} "
              f"{np.mean(nh[s]>=5):8.2f} {np.mean(nh[s]>=10):9.2f}")

    for label, cset in [("pions", {3, 4}), ("protons", {7}), ("muons", {13, 14})]:
        s = np.isin(cls, list(cset))
        if s.sum() < 500:
            continue
        print(f"\n2-3. within {label} (n={s.sum()}): corr(n_hits, .)")
        print(f"     incidence={np.corrcoef(nh[s], inc[s])[0,1]:+.3f}  logE={np.corrcoef(nh[s], logE[s])[0,1]:+.3f}  "
              f"|eta|={np.corrcoef(nh[s], aeta[s])[0,1]:+.3f}  log pT={np.corrcoef(nh[s], np.log(pt[s]))[0,1]:+.3f}")
        # n_hits vs incidence quartiles
        q = np.quantile(inc[s], [0, .25, .5, .75, 1.0]); row = []
        for k in range(4):
            mm = s & (inc >= q[k]) & (inc <= q[k+1] if k == 3 else inc < q[k+1])
            row.append(f"[{q[k]:.0f}-{q[k+1]:.0f}deg]:{nh[mm].mean():.1f}")
        print(f"     n_hits vs incidence: " + "  ".join(row))

    print("\n4. DEPTH (energy-loss proxy): per-layer survival among pions")
    s = np.isin(cls, [3, 4]); nhp = nh[s]
    prev = len(nhp)
    for L in range(1, 11):
        reach = np.mean(nhp >= L); cont = np.mean(nhp >= L + 1)
        surv = cont / max(reach, 1e-9)
        print(f"   reached hit {L:2d}: {reach:.3f} of pions   P(survive to {L+1})={surv:.3f}")


if __name__ == "__main__":
    main()
