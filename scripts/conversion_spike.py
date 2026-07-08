"""Cascade spike: train + validate a photon->conversion generator.

Conditioned ONLY on the photon [log_E, eta, vr, vz] (noise-free input), three heads:
  convert  : Bernoulli(converts?)
  radius   : Gaussian(log conversion radius | converted)
  esplit   : Gaussian(leading-e± energy fraction | >=2 e daughters)
Validates generated marginals vs truth (convert fraction, conv radius, e-split) and
the KEY conditional convert-fraction-vs-photon-energy (the 89%->10% curve): can the
model reproduce the tracker-vs-calo split from conditioning alone?
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            L.append(nn.SiLU())
    return nn.Sequential(*L)


class ConvModel(nn.Module):
    def __init__(self, norm, embed=64):
        super().__init__()
        self.enc = mlp([4, embed, embed])
        self.convert = nn.Linear(embed, 1)
        self.radius = nn.Linear(embed, 2)   # mu, log_sigma
        self.split = nn.Linear(embed, 2)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def heads(self, cond):
        h = self.enc((cond - self.cond_mean) / self.cond_std)
        r = self.radius(h); s = self.split(h)
        return self.convert(h)[:, 0], (r[:, 0], r[:, 1].clamp(-4, 3)), (s[:, 0], s[:, 1].clamp(-4, 3))


def gnll(x, mu, ls):
    return 0.5 * (((x - mu) / ls.exp()) ** 2 + 2 * ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/photon_conv.npz")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/photon_conv_eval.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); rng = np.random.default_rng(0)

    d = np.load(args.slice)
    cond = d["cond"]; conv = d["converted"]; lcr = d["log_conv_r"]; esp = d["esplit"]
    norm = {"cond_mean": d["cond_mean"], "cond_std": d["cond_std"]}
    S = len(cond)
    perm = rng.permutation(S); nval = S // 10
    val = perm[:nval]; tr = perm[nval:]

    model = ConvModel(norm).to(dev)
    C = torch.as_tensor(cond, device=dev)
    CV = torch.as_tensor(conv, device=dev)
    LR = torch.as_tensor(np.nan_to_num(lcr), device=dev); has_r = torch.as_tensor(~np.isnan(lcr), device=dev)
    ES = torch.as_tensor(np.nan_to_num(esp), device=dev); has_s = torch.as_tensor(~np.isnan(esp), device=dev)
    tri = torch.as_tensor(tr, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, args.steps + 1):
        bi = tri[torch.randint(len(tri), (args.batch,), device=dev)]
        clog, (rmu, rls), (smu, sls) = model.heads(C[bi])
        loss = F.binary_cross_entropy_with_logits(clog, CV[bi])
        mr = has_r[bi]
        if mr.any():
            loss = loss + (gnll(LR[bi], rmu, rls) * mr).sum() / mr.sum()
        ms = has_s[bi]
        if ms.any():
            loss = loss + (gnll(ES[bi], smu, sls) * ms).sum() / ms.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  step {step} | loss {loss.item():.4f}", flush=True)
    ckpt = Path(args.out).with_name("photon_conv_model.pt")
    torch.save({"model": model.state_dict(), "norm": norm}, ckpt)
    print(f"saved {ckpt}")

    # ---- validate on held-out photons ----
    model.eval()
    with torch.no_grad():
        clog, (rmu, rls), (smu, sls) = model.heads(C[torch.as_tensor(val, device=dev)])
        p = torch.sigmoid(clog)
        g_conv = (torch.rand_like(p) < p).cpu().numpy()
        g_lcr = (rmu + torch.randn_like(rmu) * rls.exp()).cpu().numpy()
        g_esp = (smu + torch.randn_like(smu) * sls.exp()).clamp(0.5, 1.0).cpu().numpy()
    t_conv = conv[val] > 0.5
    t_lcr = lcr[val]; t_esp = esp[val]

    print("\n" + "=" * 56); print("CONVERSION SPIKE — validation (held-out photons)"); print("=" * 56)
    print(f"convert fraction:   truth {t_conv.mean():.1%}   gen {g_conv.mean():.1%}")
    tr_r = t_lcr[~np.isnan(t_lcr)]; gr_r = g_lcr[g_conv]
    print(f"log conv_r:  truth mean {tr_r.mean():.2f} std {tr_r.std():.2f}   "
          f"gen mean {gr_r.mean():.2f} std {gr_r.std():.2f}")
    print(f"  (conv_r median mm: truth {np.exp(np.median(tr_r)):.0f}  gen {np.exp(np.median(gr_r)):.0f})")
    tr_s = t_esp[~np.isnan(t_esp)]; gr_s = g_esp[g_conv]
    print(f"e-split:     truth mean {tr_s.mean():.3f} std {tr_s.std():.3f}   "
          f"gen mean {gr_s.mean():.3f} std {gr_s.std():.3f}")
    print("\nconvert fraction vs photon energy (the 89%->10% curve):")
    logE = cond[val, 0]; eb = np.quantile(logE, np.linspace(0, 1, 6))
    rows = []
    for b in range(5):
        m = (logE >= eb[b]) & (logE < eb[b + 1] + (1e-6 if b == 4 else 0))
        tv, gv = t_conv[m].mean(), g_conv[m].mean()
        rows.append({"E_GeV_lo": float(np.exp(eb[b])), "truth": float(tv), "gen": float(gv)})
        print(f"  E>[{np.exp(eb[b]):7.3f}) GeV : truth {tv:5.1%}  gen {gv:5.1%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "convert_frac": {"truth": float(t_conv.mean()), "gen": float(g_conv.mean())},
        "log_conv_r": {"truth_mean": float(tr_r.mean()), "gen_mean": float(gr_r.mean()),
                       "truth_std": float(tr_r.std()), "gen_std": float(gr_r.std())},
        "esplit": {"truth_mean": float(tr_s.mean()), "gen_mean": float(gr_s.mean())},
        "convert_vs_E": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
