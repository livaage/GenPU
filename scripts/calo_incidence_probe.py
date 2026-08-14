"""Does a particle deposit in the calo at ALL? (the missing head)

Every calo slice, model and gate so far is conditioned on `n_cells >= 1` — the model is
P(shower | particle deposits), never P(deposits | particle). But only 68.7% of particles
deposit, and the rate is strongly species-dependent (e- 50%, e+ 95%, n 35%, K0L 100%), so a
full-event generator has no way to decide WHICH particles get a shower. This is the calo
analogue of the tracker's count head.

Probe: how predictable is the deposit flag from the conditioning contract (7 CONT_FEATURES +
PDG class)? A logistic regression gives the floor; a small MLP gives the achievable AUC.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CLASS_NAME = {0: "e-", 1: "e+", 2: "gamma", 3: "pi+", 4: "pi-", 5: "K+", 6: "K-", 7: "p",
              8: "pbar", 9: "n", 10: "nbar", 11: "mu-", 12: "mu+", 13: "K0L", 14: "K0S",
              15: "pi0", 16: "other"}


def rank_auc(s, y):
    o = np.argsort(s); ra = np.empty_like(o, float); ra[o] = np.arange(1, len(s) + 1)
    p = y == 1; npo, nne = p.sum(), (~p).sum()
    return 0.5 if npo == 0 or nne == 0 else (ra[p].sum() - npo * (npo + 1) / 2) / (npo * nne)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--shard", type=int, default=5)
    ap.add_argument("--n", type=int, default=1500000, help="subsample particles")
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics")
    args = ap.parse_args()
    rng = np.random.default_rng(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, off = d["particle_features"], d["particle_aux"], d["calo_offsets"]
    dep = (np.diff(off) > 0).astype(np.float32)
    idx = rng.choice(len(pf), min(args.n, len(pf)), replace=False)
    pf, aux, dep = pf[idx], aux[idx], dep[idx]
    vr = np.hypot(aux[:, AUX_VX], aux[:, AUX_VY])
    logE = np.log(np.clip(aux[:, AUX_ENERGY], 1e-6, None))
    X = np.stack([pf[:, PF_LOGPT], pf[:, PF_ETA], logE, pf[:, PF_CHARGE], pf[:, PF_MASS],
                  vr, aux[:, AUX_VZ]], 1).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    cls = pf[:, PF_PDG].astype(np.int64)
    print(f"shard {args.shard}: {len(dep)} particles, deposit rate {dep.mean():.3f}")

    print(f"\n{'class':>6} {'name':>6} {'particles':>10} {'deposit rate':>13} {'med vr [mm]':>12} {'med vr (no dep)':>16}")
    for c in sorted(set(cls.tolist())):
        m = cls == c
        nd = ~(dep[m] > 0)
        print(f"{c:>6} {CLASS_NAME.get(c,'?'):>6} {m.sum():>10} {dep[m].mean():>13.3f} "
              f"{np.median(vr[m]):>12.1f} {np.median(vr[m][nd]) if nd.any() else float('nan'):>16.1f}")

    n_tr = int(0.7 * len(dep))
    Xt = torch.as_tensor(X, device=dev); ct = torch.as_tensor(cls, device=dev)
    yt = torch.as_tensor(dep, device=dev)
    tr, te = slice(0, n_tr), slice(n_tr, None)
    res = {"deposit_rate": float(dep.mean())}
    for name, use_pdg, hidden in [("logistic (cont only)", False, 0), ("logistic (+pdg)", True, 0),
                                  ("MLP 64x2 (+pdg)", True, 64)]:
        emb = torch.nn.Embedding(17, 8).to(dev)
        din = X.shape[1] + (8 if use_pdg else 0)
        net = (torch.nn.Linear(din, 1) if not hidden else torch.nn.Sequential(
            torch.nn.Linear(din, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(), torch.nn.Linear(hidden, 1))).to(dev)
        params = list(net.parameters()) + (list(emb.parameters()) if use_pdg else [])
        opt = torch.optim.Adam(params, lr=3e-3)
        def fwd(sl):
            f = [Xt[sl]] + ([emb(ct[sl])] if use_pdg else [])
            return net(torch.cat(f, -1)).squeeze(-1)
        for _ in range(1500):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(fwd(tr), yt[tr]).backward(); opt.step()
        with torch.no_grad():
            s = fwd(te).cpu().numpy()
        auc = rank_auc(s, dep[n_tr:])
        acc = float(((s > 0) == (dep[n_tr:] > 0)).mean())
        print(f"  {name:>22}: AUC {auc:.4f}  acc {acc:.4f}")
        res[name] = {"auc": round(float(auc), 4), "acc": round(acc, 4)}
    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    (o / "incidence_probe.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", o / "incidence_probe.json")


if __name__ == "__main__":
    main()
