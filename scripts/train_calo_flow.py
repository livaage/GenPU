"""Train the CaloClouds-lite calo flow head on the photon slice (M2 spike)."""
from __future__ import annotations
import argparse, time, os
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.flow.calo_flow import CaloFlow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon.npz")
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow/photon_v1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--pt_batch", type=int, default=16384)
    ap.add_argument("--glob_batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--glob_weight", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--run_name", default="calo_flow_photon_v1")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std", "glob_mean", "glob_std", "pts_mean", "pts_std"]}
    cont, pdg, glob, pts, off = d["cont"], d["pdg"], d["glob"], d["points_flat"], d["offsets"]
    S = cont.shape[0]

    # deterministic shower-level train/val split
    rng = np.random.default_rng(0)
    perm = rng.permutation(S)
    n_val = int(S * args.val_frac)
    val_sh = np.zeros(S, bool); val_sh[perm[:n_val]] = True
    tr_sh = ~val_sh

    npt = np.diff(off)
    point_shower = np.repeat(np.arange(S), npt)          # (P,) shower id per point
    tr_pt = tr_sh[point_shower]

    model = CaloFlow(norm).to(dev)

    # standardise & move to GPU
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg, dtype=torch.long, device=dev)
    globS = torch.as_tensor((glob - norm["glob_mean"]) / norm["glob_std"], dtype=torch.float32, device=dev)
    ptsS = torch.as_tensor((pts - norm["pts_mean"]) / norm["pts_std"], dtype=torch.float32, device=dev)
    psh = torch.as_tensor(point_shower, device=dev)
    pt_contS, pt_pdg, pt_globS = contS[psh], pdgT[psh], globS[psh]
    tr_pt_idx = torch.as_tensor(np.where(tr_pt)[0], device=dev)
    tr_sh_idx = torch.as_tensor(np.where(tr_sh)[0], device=dev)
    va_pt_idx = torch.as_tensor(np.where(~tr_pt)[0], device=dev)
    va_sh_idx = torch.as_tensor(np.where(val_sh)[0], device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="genpu-caloflow", name=args.run_name, config=vars(args), mode="offline")
        except Exception as e:
            print("wandb off:", e); use_wandb = False

    print(f"device={dev}  showers={S} (train {tr_sh.sum()}, val {n_val})  points={pts.shape[0]}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    def val_loss():
        model.eval()
        with torch.no_grad():
            pi = va_pt_idx[torch.randint(len(va_pt_idx), (args.pt_batch,), device=dev)]
            ce = model.cond_embed(pt_contS[pi], pt_pdg[pi])
            cfm = model.points.cfm_loss(ptsS[pi], ce, pt_globS[pi]).item()
            si = va_sh_idx[torch.randint(len(va_sh_idx), (args.glob_batch,), device=dev)]
            g = model.glob.nll(model.cond_embed(contS[si], pdgT[si]), globS[si]).item()
        model.train()
        return cfm, g

    t0 = time.time()
    for step in range(1, args.steps + 1):
        pi = tr_pt_idx[torch.randint(len(tr_pt_idx), (args.pt_batch,), device=dev)]
        si = tr_sh_idx[torch.randint(len(tr_sh_idx), (args.glob_batch,), device=dev)]
        cfm = model.points.cfm_loss(ptsS[pi], model.cond_embed(pt_contS[pi], pt_pdg[pi]), pt_globS[pi])
        gnll = model.glob.nll(model.cond_embed(contS[si], pdgT[si]), globS[si])
        loss = cfm + args.glob_weight * gnll
        opt.zero_grad(); loss.backward(); opt.step()

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            vc, vg = val_loss()
            print(f"  step {step:6d} | cfm {cfm.item():.4f} | gnll {gnll.item():.4f} "
                  f"| val_cfm {vc:.4f} val_gnll {vg:.4f} | {sps:.1f} it/s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"train/cfm": cfm.item(), "train/gnll": gnll.item(),
                           "val/cfm": vc, "val/gnll": vg}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = Path(args.out) / f"checkpoint_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "norm": norm, "args": vars(args), "step": step}, ckpt)
            print(f"  saved {ckpt}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
