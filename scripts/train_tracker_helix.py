"""Train the v4 helix-anchored AR tracker. Slice hits = [layer, dev_x, dev_y, dev_z, time];
helix reference (B,48,3) recomputed per batch from cont + rotated vertex."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_helix_model import TrackerHelixModel

DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default=f"{DATA}/tracker_slice/helix_pion.npz")
    ap.add_argument("--out", default=f"{DATA}/checkpoints/tracker/helix_pion_v4")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=20000)
    ap.add_argument("--run_name", default="tracker_helix_v4")
    ap.add_argument("--max_particles", type=int, default=0)
    ap.add_argument("--init_from", default="")
    ap.add_argument("--ss_prob", type=float, default=0.0)
    ap.add_argument("--ss_warmup", type=int, default=0)
    ap.add_argument("--ss_ramp", type=int, default=1)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)

    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std"]}
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    vxr, vyr = d["vxr"], d["vyr"]
    if args.max_particles and args.max_particles < cont.shape[0]:
        S0 = args.max_particles
        cont, pdg, vxr, vyr = cont[:S0], pdg[:S0], vxr[:S0], vyr[:S0]
        hits = hits[:off[S0]]; off = off[:S0 + 1]
    S = cont.shape[0]; M = args.max_hits

    # precompute helix reference (S,48,3) in chunks
    href = np.zeros((S, 48, 3), np.float32)
    for s in range(0, S, 200000):
        e = min(s + 200000, S)
        href[s:e] = TrackerHelixModel.helix_ref(cont[s:e], vxr[s:e], vyr[s:e])
    # time standardization
    t_all = hits[:, 4]; t_mean, t_std = float(t_all.mean()), float(t_all.std() + 1e-6)

    rng = np.random.default_rng(0); perm = rng.permutation(S)
    n_val = int(S * args.val_frac); val_p = np.zeros(S, bool); val_p[perm[:n_val]] = True
    tr_idx = np.where(~val_p)[0]; va_idx = np.where(val_p)[0]
    contS = ((cont - norm["cont_mean"]) / norm["cont_std"]).astype(np.float32)
    lay_all = hits[:, 0].astype(np.int64); dev_all = hits[:, 1:4].astype(np.float32)
    tstd_all = ((hits[:, 4] - t_mean) / t_std).astype(np.float32)

    model = TrackerHelixModel(norm, max_hits=M).to(dev)
    model.tracker.time_mean.fill_(t_mean); model.tracker.time_std.fill_(t_std)
    if args.init_from:
        miss = model.load_state_dict(torch.load(args.init_from, map_location=dev)["model"], strict=False)
        print(f"init_from: missing={list(miss.missing_keys)[:4]}...", flush=True)

    def ss_prob_at(step):
        if args.ss_prob <= 0:
            return 0.0
        return args.ss_prob * float(np.clip((step - args.ss_warmup) / max(args.ss_ramp, 1), 0, 1))

    def make_batch(idx):
        B = len(idx)
        nh = np.minimum(off[idx + 1] - off[idx], M).astype(np.int64)
        layer = np.zeros((B, M), np.int64); devv = np.zeros((B, M, 3), np.float32)
        tt = np.zeros((B, M), np.float32); mask = np.zeros((B, M), bool)
        for j, i in enumerate(idx):
            a = off[i]; n = nh[j]
            layer[j, :n] = lay_all[a:a + n]; devv[j, :n] = dev_all[a:a + n]
            tt[j, :n] = tstd_all[a:a + n]; mask[j, :n] = True
        return (torch.as_tensor(layer, device=dev), torch.as_tensor(devv, device=dev),
                torch.as_tensor(tt, device=dev), torch.as_tensor(mask, device=dev),
                torch.as_tensor(nh, device=dev),
                torch.as_tensor(contS[idx], device=dev), torch.as_tensor(pdg[idx], dtype=torch.long, device=dev),
                torch.as_tensor(href[idx], device=dev))

    def step_loss(idx, ssp=0.0):
        layer, devv, tt, mask, nh, cS, pg, hr = make_batch(idx)
        ce = model.cond_embed(cS, pg)
        return model.tracker.loss(layer, devv, tt, ce, mask, nh, hr, ss_prob=ssp)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb; wandb.init(project="genpu-tracker", name=args.run_name, config=vars(args), mode="offline")
        except Exception as e:
            print("wandb off:", e); use_wandb = False

    nph = off[1:] - off[:-1]
    print(f"device={dev} particles={S} (train {len(tr_idx)})  hits/particle mean={nph.mean():.1f}  "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    def val_loss():
        model.eval()
        with torch.no_grad():
            out = step_loss(va_idx[np.random.randint(len(va_idx), size=min(args.batch, len(va_idx)))])
        model.train()
        return out["tracker_layer_loss"].item(), out["tracker_dev_loss"].item()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = tr_idx[np.random.randint(len(tr_idx), size=args.batch)]
        ssp = ss_prob_at(step)
        out = step_loss(idx, ssp)
        loss = out["tracker_ar_loss"]
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0:
            vl, vd = val_loss()
            print(f"  step {step:6d} | ss {ssp:.2f} | ar {loss.item():.4f} | layer {out['tracker_layer_loss'].item():.4f} "
                  f"dev {out['tracker_dev_loss'].item():.4f} time {out['tracker_time_loss'].item():.4f} "
                  f"| val_layer {vl:.4f} val_dev {vd:.4f} | {step/(time.time()-t0):.1f} it/s", flush=True)
            if use_wandb:
                import wandb; wandb.log({"train/ar": loss.item(), "train/layer": out["tracker_layer_loss"].item(),
                                        "train/dev": out["tracker_dev_loss"].item(), "val/layer": vl, "val/dev": vd}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = Path(args.out) / f"checkpoint_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "norm": norm, "args": vars(args), "step": step}, ckpt)
            print(f"  saved {ckpt}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
