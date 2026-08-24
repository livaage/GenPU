"""Which subsystem does each particle show up in, and how predictable is that?

Three measurements, all on REAL data with no model:

  A  ZERO-TRACE CENSUS (preprocessed stage2) — P(0 tracker hits), P(0 calo cells), and the
     tracker-only / calo-only / both partition. Every calo slice and every tracker slice
     filters to >=1 hit (build_count_slice_stage2.py `keep = nh >= 1`; build_tracker_slice.py
     `--min_hits 1`), and CountHead is a categorical over 1..48, so BOTH subsystems model
     P(response | particle, n >= 1) and neither has an incidence head.

  B  DETERMINISM — how much of the tracker-only/calo-only/both outcome is a function of truth
     kinematics (log pt, eta, charge, primary, vr, |vz|, log E, pdg class)? Shallow decision
     tree vs the majority-class baseline. If a small tree saturates, the residual is physical
     stochasticity (conversion / nuclear interaction), not missing features.

  C  NEUTRAL VISIBILITY MECHANISM (raw HF source) — a neutral can only touch the tracker by
     converting / decaying / interacting. Conditions P(tracker hits) on whether the particle
     has a recorded CHARGED daughter.

NOTE ON JOINS: the raw `particles` and `tracker_hits` shards are NOT in the same row order —
row-index agreement is 0.000. Join by `event_id` (see CLAUDE.md). Measurement C carries its own
wrong-event null so a broken join cannot be mistaken for signal.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
NAMES = {0: "e-", 1: "e+", 2: "gamma", 3: "pi+", 4: "pi-", 5: "K+", 6: "K-", 7: "p", 8: "pbar",
         9: "n", 10: "nbar", 11: "mu-", 12: "mu+", 13: "K0L", 14: "K0S", 15: "pi0", 16: "other"}
NEUTRAL_PDG = {22: "gamma", 2112: "n", -2112: "nbar", 130: "K0L", 310: "K0S", 111: "pi0"}


def census(stage2):
    d = np.load(stage2)
    pf, aux = d["particle_features"], d["particle_aux"]
    nt = np.diff(d["tracker_offsets"]).astype(np.int64)
    nc = np.diff(d["calo_offsets"]).astype(np.int64)
    pt = np.exp(pf[:, PF_LOGPT]); q = pf[:, PF_CHARGE]
    cls = pf[:, PF_PDG].astype(np.int64); prim = aux[:, AUX_PRIMARY] > 0.5
    vr = np.hypot(aux[:, AUX_VX], aux[:, AUX_VY]); vz = np.abs(aux[:, AUX_VZ])
    # NOTE: preprocessing.py already applied visible_mask = (n_trk > 0) | (n_cal > 0), so these
    # are rates among particles visible SOMEWHERE. Fully invisible particles are not in the file.
    y = np.where((nt >= 1) & (nc == 0), 0, np.where((nt == 0) & (nc >= 1), 1, 2))
    print(f"N = {len(y):,} (visible-somewhere)   tracker 0-hit {np.mean(nt==0):.3f}   "
          f"calo 0-cell {np.mean(nc==0):.3f}")
    print(f"  trk-only {np.mean(y==0):.3f}   calo-only {np.mean(y==1):.3f}   both {np.mean(y==2):.3f}\n")

    print(f"{'species':>7} {'frac_all':>9} {'trk-only':>9} {'calo-only':>10} {'both':>7} "
          f"{'PURITY':>7} {'med pT':>7} {'med vr':>7}")
    for c in sorted(set(cls.tolist())):
        m = cls == c
        if m.sum() < 2000:
            continue
        f = [np.mean(y[m] == k) for k in range(3)]
        print(f"{NAMES[c]:>7} {m.mean():9.3f} {f[0]:9.3f} {f[1]:10.3f} {f[2]:7.3f} "
              f"{max(f):7.3f} {np.median(pt[m]):7.3f} {np.median(vr[m]):7.1f}")

    print("\nphysical rules:")
    for lab, m in [("neutral (q==0)", q == 0),
                   ("charged, vr < 50mm", (q != 0) & (vr < 50)),
                   ("charged, vr 50-1100mm", (q != 0) & (vr >= 50) & (vr < 1100)),
                   ("charged, vr > 1100mm", (q != 0) & (vr >= 1100)),
                   ("charged primary, pT > 1GeV", (q != 0) & prim & (pt > 1))]:
        if m.sum() < 100:
            continue
        f = [np.mean(y[m] == k) for k in range(3)]
        out = ["trk-only", "calo-only", "both"][int(np.argmax(f))]
        print(f"  {lab:28s} n={m.sum():>9,}  -> {out:9s} purity {max(f):.3f}")
    return y, np.column_stack([pf[:, PF_LOGPT], pf[:, PF_ETA], q, prim.astype(np.float32),
                               vr, vz, np.log1p(aux[:, AUX_ENERGY]), cls]).astype(np.float32)


def determinism(y, X, n_sub=500_000, seed=0):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    idx = np.random.default_rng(seed).choice(len(y), min(n_sub, len(y)), replace=False)
    Xtr, Xte, Ytr, Yte = train_test_split(X[idx], y[idx], test_size=0.3, random_state=seed)
    print(f"\nmajority-class baseline {max(np.mean(Yte == k) for k in range(3)):.4f}")
    for depth in (2, 4, 8, 16):
        t = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(Xtr, Ytr)
        print(f"  tree depth {depth:>2}  test accuracy {t.score(Xte, Yte):.4f}")
    t = DecisionTreeClassifier(max_depth=8, random_state=seed).fit(Xtr, Ytr)
    fn = ["log_pt", "eta", "charge", "primary", "vr", "|vz|", "log_E", "pdg_class"]
    print("  depth-8 importance: " + "  ".join(
        f"{n} {v:.2f}" for n, v in sorted(zip(fn, t.feature_importances_), key=lambda z: -z[1])
        if v > 0.01))


def neutral_mechanism(shard, n_events):
    from genpu.data import load_shard
    t, h = load_shard("particles", shard), load_shard("tracker_hits", shard)
    ev_p, ev_h = t.column("event_id").to_numpy(), h.column("event_id").to_numpy()
    # join by event_id, NEVER row index -- the two subsets are not in the same order
    rp = {int(e): i for i, e in enumerate(ev_p)}
    rh = {int(e): i for i, e in enumerate(ev_h)}
    common = np.intersect1d(ev_p, ev_h)[:n_events]
    print(f"\njoined {len(common)} events by event_id "
          f"(row-index agreement would have been {np.mean(ev_p[:len(common)]==ev_h[:len(common)]):.3f})")

    S = {}
    for k, e in enumerate(common):
        i, j = rp[int(e)], rh[int(e)]
        gp = lambda n: np.asarray(t.column(n)[i].as_py())
        P, G, C, R = gp("particle_id"), gp("pdg_id"), gp("charge"), gp("parent_id")
        if not len(P):
            continue
        own = set(np.unique(np.asarray(h.column("particle_id")[j].as_py())).tolist())
        jn = rh[int(common[(k + 137) % len(common)])]
        null = set(np.unique(np.asarray(h.column("particle_id")[jn].as_py())).tolist())
        order = np.argsort(R); Rs = R[order]
        for lab, m in [("CHARGED", C != 0)] + [(v, G == p) for p, v in NEUTRAL_PDG.items()]:
            idx = np.where(m)[0]
            if not len(idx):
                continue
            d = S.setdefault(lab, dict(n=0, trk=0, nul=0, kid=0, trk_kid=0, trk_nokid=0))
            for r_ in idx:
                p = P[r_]
                lo, hi = np.searchsorted(Rs, p, "left"), np.searchsorted(Rs, p, "right")
                ck = order[lo:hi][C[order[lo:hi]] != 0]
                ht = p in own
                d["n"] += 1; d["trk"] += ht; d["nul"] += p in null; d["kid"] += len(ck) > 0
                d["trk_kid"] += ht and len(ck) > 0
                d["trk_nokid"] += ht and len(ck) == 0

    print(f"\n{'population':>10} {'n':>8} {'P(trk)':>8} {'NULL':>7} {'P(chgKid)':>10} "
          f"{'P(trk|kid)':>11} {'P(trk|noKid)':>13}")
    for lab, d in sorted(S.items(), key=lambda z: -z[1]["n"]):
        if d["n"] < 200:
            continue
        print(f"{lab:>10} {d['n']:>8,} {d['trk']/d['n']:>8.3f} {d['nul']/d['n']:>7.3f} "
              f"{d['kid']/d['n']:>10.3f} {d['trk_kid']/max(d['kid'],1):>11.3f} "
              f"{d['trk_nokid']/max(d['n']-d['kid'],1):>13.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed/"
                                        "shard_0000_stage2.npz")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_events", type=int, default=300)
    ap.add_argument("--skip_raw", action="store_true", help="census + determinism only")
    a = ap.parse_args()
    y, X = census(a.stage2)
    determinism(y, X)
    if not a.skip_raw:
        neutral_mechanism(a.shard, a.n_events)
