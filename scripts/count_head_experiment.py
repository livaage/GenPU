"""Conditioned count head P(n_hits | particle [, d0]). Trains WITH and WITHOUT d0 to quantify
what the impact parameter adds, and verifies the marginal + the n_hits-vs-(d0, pT, eta) dependence.
Saves the with-d0 checkpoint for the honest full-pipeline gate."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.count_head import CountHead, MAXB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/count.npz")
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker/count_head.pt")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(0)
    d = np.load(args.slice)
    cont, pdg = d["cont"], d["pdg"]
    nh = d["n_hits"] if "n_hits" in d else np.diff(d["offsets"])
    has_d0 = "d0" in d
    d0 = d["d0"] if has_d0 else np.zeros(len(nh), np.float32)
    cm, cs = d["cont_mean"], d["cont_std"]
    print(f"slice has d0={has_d0}   PIONS n_hits median={np.median(nh[np.isin(pdg,[3,4])]):.0f} mean={nh[np.isin(pdg,[3,4])].mean():.2f}")
    S = len(nh); tr = rng.permutation(S)[: int(S * 0.9)]; te = np.setdiff1d(np.arange(S), tr)
    contS = torch.as_tensor((cont - cm) / cs, dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg, dtype=torch.long, device=dev)
    d0T = torch.as_tensor(d0, dtype=torch.float32, device=dev)
    nhT = torch.as_tensor(np.clip(nh, 1, None), device=dev)
    trT = torch.as_tensor(tr, device=dev); teT = torch.as_tensor(te, device=dev)

    def train(use_d0):
        m = CountHead(use_d0=use_d0).to(dev)
        if has_d0:
            m.d0_mean[0] = float(d["d0_mean"][0]); m.d0_std[0] = float(d["d0_std"][0])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for step in range(1, args.steps + 1):
            idx = trT[torch.randint(len(trT), (args.batch,), device=dev)]
            loss = m.loss(contS[idx], pdgT[idx], d0T[idx], nhT[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nll = m.loss(contS[teT], pdgT[teT], d0T[teT], nhT[teT]).item()
        return m, nll

    m0, nll0 = train(False)
    if has_d0:
        m1, nll1 = train(True)
        print("=" * 60); print("COUNT HEAD: held-out NLL"); print("=" * 60)
        print(f"  without d0: {nll0:.4f}    with d0: {nll1:.4f}    improvement: {nll0-nll1:+.4f}")
    else:
        m1, nll1 = m0, nll0
        print(f"COUNT HEAD (no d0 in slice): held-out NLL {nll0:.4f}")

    real = nh[te]
    gen = m1.sample(contS[teT], pdgT[teT], d0T[teT]).cpu().numpy()
    print("\nmarginal (real vs with-d0 sampled):")
    for k in [1, 2, 3, 5, 10]:
        print(f"  frac[=={k}]: real {np.mean(real==k):.3f} gen {np.mean(gen==k):.3f}   "
              f"[>={k}]: real {np.mean(real>=k):.3f} gen {np.mean(gen>=k):.3f}")
    print(f"  mean real {real.mean():.2f} gen {gen.mean():.2f}")
    print("\nn_hits vs bins (real / pred-no-d0 / pred-with-d0):")
    e0 = m0.expected(contS[teT], pdgT[teT], d0T[teT]).cpu().numpy()
    e1 = m1.expected(contS[teT], pdgT[teT], d0T[teT]).cpu().numpy()
    for name, v in [("|d0|", np.abs(d0[te])), ("log_pt", cont[te, 0]), ("|eta|", np.abs(cont[te, 1]))]:
        q = np.quantile(v, [0, .25, .5, .75, 1.0]); rr, g0, g1 = [], [], []
        for i in range(4):
            b = (v >= q[i]) & (v <= q[i+1] if i == 3 else v < q[i+1])
            rr.append(f"{real[b].mean():.1f}"); g0.append(f"{e0[b].mean():.1f}"); g1.append(f"{e1[b].mean():.1f}")
        print(f"  {name:8s} real:{' '.join(rr)}  noD0:{' '.join(g0)}  withD0:{' '.join(g1)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m1.state_dict()}, args.out)
    print(f"\nsaved with-d0 count head -> {args.out}")


if __name__ == "__main__":
    main()
