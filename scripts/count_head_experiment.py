"""Conditioned count head: predict P(n_hits | particle) as a categorical, trained on the
per-particle tracker response (ALL particles — debris included; this is a pileup generator,
not a track reconstructor). Verify it reproduces BOTH the n_hits MARGINAL (the 43%-single-hit
shape) and the weak-but-real CONDITIONING (n_hits vs incidence angle / eta / pT) we measured.
No truth n_hits at generation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.conditioning import ParticleConditioning
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS

MAXB = 48   # n_hits bins: 1..47, 48+ overflow


class CountHead(nn.Module):
    def __init__(self, embed_dim=64, hidden=128):
        super().__init__()
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=True)
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, MAXB))

    def logits(self, cont_std, pdg):
        return self.net(self.cond(cont_std, pdg))

    def loss(self, cont_std, pdg, n_hits):
        tgt = torch.clamp(n_hits - 1, 0, MAXB - 1)
        return F.cross_entropy(self.logits(cont_std, pdg), tgt)

    @torch.no_grad()
    def sample(self, cont_std, pdg):
        return torch.distributions.Categorical(logits=self.logits(cont_std, pdg)).sample() + 1

    @torch.no_grad()
    def expected(self, cont_std, pdg):
        p = torch.softmax(self.logits(cont_std, pdg), -1)
        return (p * (torch.arange(MAXB, device=p.device) + 1)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(0)
    d = np.load(args.slice)
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    cm, cs = d["cont_mean"], d["cont_std"]
    n_hits = np.clip(np.diff(off), 1, None).astype(np.int64)
    S = cont.shape[0]
    # incidence at first hit (from measured first-hit pos vs vertex), for the verification
    lc = hits[:, 0].astype(int).clip(0, N_LAYERS - 1)
    r = hits[:, 1] * LAYER_STDS[lc, 0] + LAYER_MEANS[lc, 0]
    phi = hits[:, 2] * LAYER_STDS[lc, 1] + LAYER_MEANS[lc, 1]
    z = hits[:, 3] * LAYER_STDS[lc, 2] + LAYER_MEANS[lc, 2]
    fx, fy, fz = (r * np.cos(phi))[off[:-1]], (r * np.sin(phi))[off[:-1]], z[off[:-1]]
    vx = cont[:, 5] * np.cos(0); vz = cont[:, 6]         # vr along x (phi arbitrary), vz
    dvec = np.stack([fx - vx, fy, fz - vz], 1); dvec /= np.linalg.norm(dvec, axis=1, keepdims=True) + 1e-9
    rhat = np.stack([fx, fy, np.zeros_like(fz)], 1); rhat /= np.linalg.norm(rhat, axis=1, keepdims=True) + 1e-9
    inc = np.degrees(np.arccos(np.clip(np.abs((dvec * rhat).sum(1)), 0, 1)))

    tr = rng.permutation(S)[: int(S * 0.9)]; te = np.setdiff1d(np.arange(S), tr)
    contS = torch.as_tensor((cont - cm) / cs, dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg, dtype=torch.long, device=dev)
    nhT = torch.as_tensor(n_hits, device=dev)
    m = CountHead().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    trT = torch.as_tensor(tr, device=dev)
    for step in range(1, args.steps + 1):
        idx = trT[torch.randint(len(trT), (args.batch,), device=dev)]
        loss = m.loss(contS[idx], pdgT[idx], nhT[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  step {step} loss {loss.item():.3f}", flush=True)

    teT = torch.as_tensor(te, device=dev)
    gen = m.sample(contS[teT], pdgT[teT]).cpu().numpy(); real = n_hits[te]
    print("=" * 60); print("COUNT HEAD — marginal (real vs sampled)"); print("=" * 60)
    for k in [1, 2, 3, 5, 10]:
        print(f"  frac[=={k}]: real {np.mean(real==k):.3f}  gen {np.mean(gen==k):.3f}   "
              f"frac[>={k}]: real {np.mean(real>=k):.3f}  gen {np.mean(gen>=k):.3f}")
    print(f"  mean: real {real.mean():.2f}  gen {gen.mean():.2f}   median: real {np.median(real):.0f} gen {np.median(gen):.0f}")
    print("\nCONDITIONING captured? mean n_hits in bins (real vs count-head expected):")
    exp = m.expected(contS[teT], pdgT[teT]).cpu().numpy()
    for name, v in [("incidence", inc[te]), ("|eta|", np.abs(cont[te, 1])), ("log_pt", cont[te, 0])]:
        q = np.quantile(v, [0, .25, .5, .75, 1.0]); rr, gg = [], []
        for i in range(4):
            b = (v >= q[i]) & (v <= q[i+1] if i == 3 else v < q[i+1])
            rr.append(f"{real[b].mean():.1f}"); gg.append(f"{exp[b].mean():.1f}")
        print(f"  {name:10s} real: {' '.join(rr)}   pred: {' '.join(gg)}")


if __name__ == "__main__":
    main()
