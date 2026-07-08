"""Cascade hard-case spike: train + validate a hadron nuclear-interaction generator.

Conditioned ONLY on the hadron [log_E, eta, vr, vz], four heads:
  interact : Bernoulli(interacts?)
  radius   : Gaussian(log interaction radius | interacted)
  mult     : Categorical(multiplicity 1..MCAP | interacted)   <- the new hard part
  efrac    : Gaussian(sum E_dau / E_parent | interacted)
Validates generated marginals vs truth (interact frac, radius, MULTIPLICITY dist,
efrac) and the interact-vs-energy conditional. The multiplicity distribution is the
make-or-break: can a generator reproduce the variable daughter count?
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MCAP = 20  # multiplicities >=MCAP folded into the top class


def mlp(sizes):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.SiLU())
    return nn.Sequential(*L)


class NucModel(nn.Module):
    def __init__(self, norm, embed=64):
        super().__init__()
        self.enc = mlp([4, embed, embed])
        self.interact = nn.Linear(embed, 1)
        self.radius = nn.Linear(embed, 2)
        self.mult = nn.Linear(embed, MCAP)       # classes = multiplicity 1..MCAP
        self.efrac = nn.Linear(embed, 2)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def heads(self, cond):
        h = self.enc((cond - self.cond_mean) / self.cond_std)
        r = self.radius(h); e = self.efrac(h)
        return (self.interact(h)[:, 0], (r[:, 0], r[:, 1].clamp(-4, 3)),
                self.mult(h), (e[:, 0], e[:, 1].clamp(-4, 3)))


def gnll(x, mu, ls):
    return 0.5 * (((x - mu) / ls.exp()) ** 2 + 2 * ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/pion_nuclear.npz")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/pion_nuclear_eval.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); rng = np.random.default_rng(0)

    d = np.load(args.slice)
    cond = d["cond"]; inter = d["inter"]; lir = d["log_int_r"]; mult = d["mult"]; ef = d["efrac"]
    norm = {"cond_mean": d["cond_mean"], "cond_std": d["cond_std"]}
    mult_cls = np.clip(mult, 1, MCAP) - 1     # 0..MCAP-1
    S = len(cond); perm = rng.permutation(S); nval = S // 10
    val = perm[:nval]; tr = perm[nval:]

    model = NucModel(norm).to(dev)
    C = torch.as_tensor(cond, device=dev); IN = torch.as_tensor(inter, device=dev)
    LR = torch.as_tensor(np.nan_to_num(lir), device=dev); hr = torch.as_tensor(~np.isnan(lir), device=dev)
    MC = torch.as_tensor(mult_cls, device=dev)
    EF = torch.as_tensor(np.nan_to_num(ef), device=dev); he = torch.as_tensor(~np.isnan(ef), device=dev)
    tri = torch.as_tensor(tr, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, args.steps + 1):
        bi = tri[torch.randint(len(tri), (args.batch,), device=dev)]
        ilog, (rmu, rls), mlog, (emu, els) = model.heads(C[bi])
        loss = F.binary_cross_entropy_with_logits(ilog, IN[bi])
        m = hr[bi]
        if m.any():
            loss = loss + (gnll(LR[bi], rmu, rls) * m).sum() / m.sum()
            loss = loss + F.cross_entropy(mlog[m], MC[bi][m])
            loss = loss + (gnll(EF[bi], emu, els) * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2500 == 0:
            print(f"  step {step} | loss {loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        ilog, (rmu, rls), mlog, (emu, els) = model.heads(C[torch.as_tensor(val, device=dev)])
        p = torch.sigmoid(ilog)
        g_int = (torch.rand_like(p) < p).cpu().numpy()
        g_lr = (rmu + torch.randn_like(rmu) * rls.exp()).cpu().numpy()
        g_mult = (torch.distributions.Categorical(logits=mlog).sample() + 1).cpu().numpy()
        g_ef = (emu + torch.randn_like(emu) * els.exp()).cpu().numpy()
    t_int = inter[val] > 0.5
    t_lr = lir[val]; t_mult = np.clip(mult[val], 1, MCAP); t_ef = ef[val]

    print("\n" + "=" * 56); print("NUCLEAR SPIKE — validation (held-out hadrons)"); print("=" * 56)
    print(f"interact fraction:  truth {t_int.mean():.1%}   gen {g_int.mean():.1%}")
    tr_r = t_lr[~np.isnan(t_lr)]; gr_r = g_lr[g_int]
    print(f"log int_r:  truth mean {tr_r.mean():.2f} std {tr_r.std():.2f}   gen mean {gr_r.mean():.2f} std {gr_r.std():.2f}")
    tm = t_mult[t_int]; gm = g_mult[g_int]
    print(f"MULTIPLICITY: truth mean {tm.mean():.2f} median {np.median(tm):.0f}   gen mean {gm.mean():.2f} median {np.median(gm):.0f}")
    print("  mult dist (truth / gen) for m=1..8:")
    for mm in range(1, 9):
        print(f"    m={mm}: {(tm==mm).mean():5.1%} / {(gm==mm).mean():5.1%}")
    tr_e = t_ef[~np.isnan(t_ef)]; gr_e = g_ef[g_int]
    print(f"efrac:   truth median {np.median(tr_e):.2f}   gen median {np.median(gr_e):.2f}")
    print("\ninteract fraction vs hadron energy:")
    logE = cond[val, 0]; eb = np.quantile(logE, np.linspace(0, 1, 6))
    rows = []
    for b in range(5):
        mE = (logE >= eb[b]) & (logE < eb[b + 1] + (1e-6 if b == 4 else 0))
        rows.append({"E_lo": float(np.exp(eb[b])), "truth": float(t_int[mE].mean()), "gen": float(g_int[mE].mean())})
        print(f"  E>[{np.exp(eb[b]):7.3f}) GeV : truth {t_int[mE].mean():5.1%}  gen {g_int[mE].mean():5.1%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "interact_frac": {"truth": float(t_int.mean()), "gen": float(g_int.mean())},
        "mult": {"truth_mean": float(tm.mean()), "gen_mean": float(gm.mean()),
                 "truth_dist": [float((tm == mm).mean()) for mm in range(1, 9)],
                 "gen_dist": [float((gm == mm).mean()) for mm in range(1, 9)]},
        "log_int_r": {"truth_mean": float(tr_r.mean()), "gen_mean": float(gr_r.mean())},
        "interact_vs_E": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
