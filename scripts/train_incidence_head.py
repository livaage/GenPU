"""Incidence head: P(leaves a tracker trace) and P(leaves a calo trace), per particle.

THE HOLE THIS FILLS. Every slice in this project filters to >=1 hit --
build_count_slice_stage2.py `keep = nh >= 1`, build_tracker_slice.py `--min_hits 1`,
build_calo_slice.py `>=1 calo deposit` -- and CountHead is a categorical over 1..48 that
structurally cannot emit zero. So both subsystems model P(response | particle, n >= 1) and neither
can answer P(response | particle). Measured 2026-08-24 on shard 0: 59.6% of particles leave no
tracker trace and 31.3% no calo trace, and only 9.2% are seen by both. Without this head there is
no honest full-event gate on either subsystem.

WHY THE GRAPH AND NOT STAGE2. Stage2 is the filtered set by construction -- it contains only
particles that left a trace, so the negatives are absent. `shard_XXXX_graph.npz` carries every RAW
particle with `n_tracker_hits` / `n_calo_hits`, so the labels exist.

SHAPE: one shared trunk, TWO Bernoulli heads. The two outcomes are not independent (9.2% hit both),
but ~94% of the 3-way outcome is predictable from truth kinematics, so most of the dependence is
explained by the conditioning rather than needing an explicit joint. The shared trunk carries what
is left. Validation checks the 3-WAY joint fractions, not just the marginals -- that is what tells
us whether two Bernoullis were enough.

CALIBRATION, NOT ACCURACY, IS THE TARGET. The residual ~5% is the conversion / interaction coin
flip (a photon touches the tracker at 0.185 if it converted and 0.001 if not), which a generator
must SAMPLE, not predict. So we report ECE and the sampled joint fractions alongside AUC; a head
that is 94% accurate but miscalibrated would produce the wrong number of tracks per event.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.preprocessing import pdg_to_class, N_PDG_CLASSES  # noqa: E402

FEATS = ["log_E", "log_pt", "eta", "charge", "log1p_vr", "abs_vz", "log1p_mass", "d0"]


class IncidenceHead(nn.Module):
    def __init__(self, n_cont=len(FEATS), embed=64, hidden=128, n_pdg=N_PDG_CLASSES):
        super().__init__()
        self.emb = nn.Embedding(n_pdg, 16)
        self.trunk = nn.Sequential(nn.Linear(n_cont + 16, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, embed), nn.SiLU())
        self.head = nn.Linear(embed, 2)          # [logit P(tracker), logit P(calo)]
        self.register_buffer("mean", torch.zeros(n_cont))
        self.register_buffer("std", torch.ones(n_cont))

    def logits(self, cont_phys, pdg):
        x = (cont_phys - self.mean) / self.std
        return self.head(self.trunk(torch.cat([x, self.emb(pdg)], -1)))

    @torch.no_grad()
    def sample(self, cont_phys, pdg):
        p = torch.sigmoid(self.logits(cont_phys, pdg))
        return (torch.rand_like(p) < p)


def build(graph_dir, shards):
    C, P, Y = [], [], []
    for sh in shards:
        g = np.load(f"{graph_dir}/shard_{sh:04d}_graph.npz")
        px, py, pz = g["px"].astype(np.float64), g["py"].astype(np.float64), g["pz"].astype(np.float64)
        pt = np.hypot(px, py); pm = np.sqrt(px**2 + py**2 + pz**2)
        eta = np.arcsinh(np.clip(pz / np.clip(pt, 1e-9, None), -30, 30))
        phi = np.arctan2(py, px)
        vr = np.hypot(g["vx"], g["vy"])
        d0 = g["perigee_d0"].astype(np.float64)
        C.append(np.stack([np.log(np.clip(g["energy"], 1e-9, None)),
                           np.log(np.clip(pt, 1e-9, None)), eta, g["charge"],
                           np.log1p(vr), np.abs(g["vz"]), np.log1p(g["mass"]), d0], 1))
        P.append(pdg_to_class(g["pdg_id"].astype(np.int64)).astype(np.int64))
        Y.append(np.stack([(g["n_tracker_hits"] > 0), (g["n_calo_hits"] > 0)], 1))
        print(f"  shard {sh}: +{len(P[-1]):,}", flush=True)
    return (np.concatenate(C).astype(np.float32), np.concatenate(P),
            np.concatenate(Y).astype(np.float32))


def ece(p, y, nb=15):
    e = 0.0
    for k in range(nb):
        m = (p >= k / nb) & (p < (k + 1) / nb)
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade_graph")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--val_shard", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/incidence")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("building train:", flush=True)
    Xc, Xp, Y = build(a.graph_dir, a.shards)
    print("building val:", flush=True)
    Vc, Vp, Vy = build(a.graph_dir, [a.val_shard])
    print(f"train {len(Y):,}  val {len(Vy):,}   base rates  tracker {Y[:,0].mean():.4f} "
          f"calo {Y[:,1].mean():.4f}", flush=True)

    m = IncidenceHead().to(dev)
    m.mean.copy_(torch.tensor(Xc.mean(0))); m.std.copy_(torch.tensor(Xc.std(0) + 1e-6))
    xc = torch.tensor(Xc, device=dev); xp = torch.tensor(Xp, device=dev); yy = torch.tensor(Y, device=dev)
    vc = torch.tensor(Vc, device=dev); vp = torch.tensor(Vp, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    for s in range(a.steps):
        i = torch.randint(len(yy), (a.batch,), device=dev)
        loss = F.binary_cross_entropy_with_logits(m.logits(xc[i], xp[i]), yy[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 2000 == 0 or s == a.steps - 1:
            print(f"  step {s:>6} | bce {loss.item():.4f}", flush=True)

    m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m.logits(vc, vp)).cpu().numpy()
        samp = m.sample(vc, vp).cpu().numpy()
    res = {}
    for j, nm in enumerate(["tracker", "calo"]):
        s_, y_ = pv[:, j], Vy[:, j]
        o = np.argsort(s_); r = np.empty_like(o, float); r[o] = np.arange(1, len(s_) + 1)
        pos = y_ == 1
        auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
        res[nm] = dict(auc=round(float(auc), 4), ece=round(ece(s_, y_), 4),
                       rate_real=round(float(y_.mean()), 4), rate_pred=round(float(s_.mean()), 4),
                       rate_sampled=round(float(samp[:, j].mean()), 4))
        print(f"{nm:>8}: AUC {auc:.4f}  ECE {res[nm]['ece']:.4f}  "
              f"rate real {y_.mean():.4f} / mean-pred {s_.mean():.4f} / SAMPLED {samp[:,j].mean():.4f}")

    print(f"\n3-WAY JOINT (the test of whether two Bernoullis suffice)")
    print(f"{'outcome':>12} {'real':>9} {'sampled':>9}")
    lab = {(1, 0): "trk-only", (0, 1): "calo-only", (1, 1): "both", (0, 0): "neither"}
    joint = {}
    for k, nm in lab.items():
        rr = float(((Vy[:, 0] == k[0]) & (Vy[:, 1] == k[1])).mean())
        gg = float(((samp[:, 0] == k[0]) & (samp[:, 1] == k[1])).mean())
        joint[nm] = dict(real=round(rr, 4), sampled=round(gg, 4))
        print(f"{nm:>12} {rr:>9.4f} {gg:>9.4f}")
    res["joint"] = joint
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({"state": m.state_dict(), "feats": FEATS}, Path(a.out) / f"incidence_s{a.seed}.pt")
    json.dump(res, open(Path(a.out) / f"incidence_s{a.seed}.json", "w"), indent=2)
    print(f"\nwrote {a.out}/incidence_s{a.seed}.pt")


if __name__ == "__main__":
    main()
