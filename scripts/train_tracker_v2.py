"""Train the v2 state-carrying AR tracker (direction state + stop head)."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_state_ar import TrackerStateModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/v2_pion.npz")
    ap.add_argument("--out", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/tracker/v2_pion_v1")
    ap.add_argument("--steps", type=int, default=70000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--run_name", default="tracker_v2_pion_v1")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std"]}
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    S, M = cont.shape[0], args.max_hits
    rng = np.random.default_rng(0)
    perm = rng.permutation(S); n_val = int(S * args.val_frac)
    val_p = np.zeros(S, bool); val_p[perm[:n_val]] = True
    tr_idx = np.where(~val_p)[0]; va_idx = np.where(val_p)[0]

    model = TrackerStateModel(norm, max_hits=M).to(dev)
    contS = ((cont - norm["cont_mean"]) / norm["cont_std"]).astype(np.float32)
    lc_all = hits[:, 0].astype(np.int64)
    resid_all = hits[:, 1:5].astype(np.float32)      # r,phi,z,time
    dir_all = hits[:, 5:7].astype(np.float32)        # theta, alpha

    def make_batch(idx):
        B = len(idx)
        n_hits = np.minimum(off[idx + 1] - off[idx], M).astype(np.int64)
        layer = np.zeros((B, M), np.int64); cont4 = np.zeros((B, M, 4), np.float32)
        direction = np.zeros((B, M, 2), np.float32); mask = np.zeros((B, M), bool)
        for j, i in enumerate(idx):
            a = off[i]; nh = n_hits[j]
            layer[j, :nh] = lc_all[a:a + nh]; cont4[j, :nh] = resid_all[a:a + nh]
            direction[j, :nh] = dir_all[a:a + nh]; mask[j, :nh] = True
        cS = torch.as_tensor(contS[idx], device=dev); pg = torch.as_tensor(pdg[idx], dtype=torch.long, device=dev)
        vtx = torch.as_tensor(cont[idx][:, [5, 6]], dtype=torch.float32, device=dev)
        return (torch.as_tensor(layer, device=dev), torch.as_tensor(cont4, device=dev),
                torch.as_tensor(direction, device=dev), torch.as_tensor(n_hits, device=dev), cS, pg, vtx)

    def step_loss(idx):
        layer, cont4, direction, nh, cS, pg, vtx = make_batch(idx)
        ce = model.cond_embed(cS, pg)
        return model.tracker.loss(layer, cont4, direction, ce, nh, vtx)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb; wandb.init(project="genpu-tracker-v2", name=args.run_name, config=vars(args), mode="offline")
        except Exception as e:
            print("wandb off:", e); use_wandb = False

    nph = off[1:] - off[:-1]
    print(f"device={dev} particles={S} (train {len(tr_idx)}, val {n_val}) hits={hits.shape[0]} "
          f"hits/p mean={nph.mean():.1f} max={nph.max()} params={sum(p.numel() for p in model.parameters()):,}")

    def val_loss():
        model.eval()
        with torch.no_grad():
            idx = va_idx[np.random.randint(len(va_idx), size=min(args.batch, len(va_idx)))]
            out = step_loss(idx)
        model.train()
        return {k: v.item() for k, v in out.items()}

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = tr_idx[np.random.randint(len(tr_idx), size=args.batch)]
        out = step_loss(idx); loss = out["total"]
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.log_every == 0:
            vl = val_loss(); sps = step / (time.time() - t0)
            print(f"  step {step:6d} | tot {loss.item():.3f} | layer {out['layer'].item():.3f} "
                  f"pos {out['pos'].item():.3f} dir {out['dir'].item():.3f} stop {out['stop'].item():.3f} "
                  f"| val_dir {vl['dir']:.3f} val_stop {vl['stop']:.3f} | {sps:.1f} it/s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({f"train/{k}": v.item() for k, v in out.items()} | {f"val/{k}": v for k, v in vl.items()}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            ck = Path(args.out) / f"checkpoint_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "norm": norm, "args": vars(args), "step": step}, ck)
            print(f"  saved {ck}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
