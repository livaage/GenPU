"""Train the v3 surface-local AR tracker (module head + local coords) on a surface slice.
Mirrors train_tracker.py; the slice hits are [module_index, x_res, y_res, z_res, time_res]."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_module_model import TrackerModuleModel

DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default=f"{DATA}/tracker_slice/surface_pion.npz")
    ap.add_argument("--module_geometry", default=f"{DATA}/module_geometry.npz")
    ap.add_argument("--out", default=f"{DATA}/checkpoints/tracker/surface_pion_v3")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--run_name", default="tracker_surface_v3")
    ap.add_argument("--max_particles", type=int, default=0, help="subsample first N particles (0=all; CPU smoke)")
    ap.add_argument("--min_hits_train", type=int, default=1)
    ap.add_argument("--init_from", default="", help="warm-start model weights from a checkpoint (strict=True)")
    ap.add_argument("--ss_prob", type=float, default=0.0, help="scheduled-sampling target prob (frac of input hits replaced by own predictions)")
    ap.add_argument("--ss_warmup", type=int, default=0, help="steps of pure teacher forcing before ss ramps in")
    ap.add_argument("--ss_ramp", type=int, default=1, help="steps to linearly ramp ss_prob from 0 to target after warmup")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std"]}
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    if args.max_particles and args.max_particles < cont.shape[0]:
        S0 = args.max_particles
        cont, pdg = cont[:S0], pdg[:S0]
        hits = hits[:off[S0]]; off = off[:S0 + 1]
    S = cont.shape[0]; M = args.max_hits

    rng = np.random.default_rng(0)
    perm = rng.permutation(S)
    n_val = int(S * args.val_frac)
    val_p = np.zeros(S, bool); val_p[perm[:n_val]] = True
    tr_idx = np.where(~val_p)[0]; va_idx = np.where(val_p)[0]
    if args.min_hits_train > 1:
        nph_ = off[1:] - off[:-1]
        tr_idx = tr_idx[nph_[tr_idx] >= args.min_hits_train]
        va_idx = va_idx[nph_[va_idx] >= args.min_hits_train]
        print(f"min_hits_train={args.min_hits_train}: train tracks {len(tr_idx)} (of {S})")

    contS = ((cont - norm["cont_mean"]) / norm["cont_std"]).astype(np.float32)
    mod_all = hits[:, 0].astype(np.int64)          # module index per hit
    resid_all = hits[:, 1:5].astype(np.float32)    # local (x,y,z,time) residuals

    model = TrackerModuleModel(norm, module_geometry_path=args.module_geometry, max_hits=M).to(dev)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=dev)["model"]
        miss = model.load_state_dict(sd, strict=False)
        print(f"init_from {args.init_from}: missing={list(miss.missing_keys)} unexpected={list(miss.unexpected_keys)}", flush=True)

    def ss_prob_at(step):
        if args.ss_prob <= 0:
            return 0.0
        return args.ss_prob * float(np.clip((step - args.ss_warmup) / max(args.ss_ramp, 1), 0.0, 1.0))

    def make_batch(idx):
        B = len(idx)
        n_hits = np.minimum(off[idx + 1] - off[idx], M).astype(np.int64)
        module = np.zeros((B, M), np.int64)
        cont4 = np.zeros((B, M, 4), np.float32)
        mask = np.zeros((B, M), bool)
        for j, i in enumerate(idx):
            a = off[i]; nh = n_hits[j]
            module[j, :nh] = mod_all[a:a + nh]
            cont4[j, :nh] = resid_all[a:a + nh]
            mask[j, :nh] = True
        cS = torch.as_tensor(contS[idx], dtype=torch.float32, device=dev)
        pg = torch.as_tensor(pdg[idx], dtype=torch.long, device=dev)
        return (torch.as_tensor(module, device=dev), torch.as_tensor(cont4, device=dev),
                torch.as_tensor(mask, device=dev), torch.as_tensor(n_hits, device=dev), cS, pg)

    def step_loss(idx, ss_prob=0.0):
        module, cont4, mask, nh, cS, pg = make_batch(idx)
        ce = model.cond_embed(cS, pg)
        return model.tracker.loss(module, cont4, ce, mask, nh, ss_prob=ss_prob)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="genpu-tracker", name=args.run_name, config=vars(args), mode="offline")
        except Exception as e:
            print("wandb off:", e); use_wandb = False

    nph = off[1:] - off[:-1]
    print(f"device={dev}  particles={S} (train {len(tr_idx)}, val {n_val})  hits={hits.shape[0]}  "
          f"hits/particle mean={nph.mean():.1f} max={nph.max()}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    def val_loss():
        model.eval()
        with torch.no_grad():
            idx = va_idx[np.random.randint(len(va_idx), size=min(args.batch, len(va_idx)))]
            out = step_loss(idx)
        model.train()
        return (out["tracker_layer_loss"].item(), out["tracker_surface_loss"].item(),
                out["tracker_continuous_loss"].item())

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = tr_idx[np.random.randint(len(tr_idx), size=args.batch)]
        ssp = ss_prob_at(step)
        out = step_loss(idx, ss_prob=ssp)
        loss = out["tracker_ar_loss"]
        opt.zero_grad(); loss.backward(); opt.step()

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            vl, vs, vc = val_loss()
            print(f"  step {step:6d} | ss {ssp:.2f} | ar {loss.item():.4f} | layer {out['tracker_layer_loss'].item():.4f} "
                  f"surf {out['tracker_surface_loss'].item():.4f} cont {out['tracker_continuous_loss'].item():.4f} "
                  f"| val_layer {vl:.4f} val_surf {vs:.4f} val_cont {vc:.4f} | {sps:.1f} it/s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"train/ar": loss.item(), "train/layer": out["tracker_layer_loss"].item(),
                           "train/surf": out["tracker_surface_loss"].item(),
                           "train/cont": out["tracker_continuous_loss"].item(),
                           "val/layer": vl, "val/surf": vs, "val/cont": vc}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = Path(args.out) / f"checkpoint_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "norm": norm, "args": vars(args), "step": step}, ckpt)
            print(f"  saved {ckpt}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
