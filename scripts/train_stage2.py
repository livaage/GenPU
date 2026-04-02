#!/usr/bin/env python
"""Train the Stage 2 per-particle detector response model.

Usage:
    python scripts/train_stage2.py [--config CONFIG]

Reads preprocessed shards from GENPU_DATA_DIR/preprocessed/.
Logs to wandb if available, otherwise tensorboard.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.dataset import ParticleHitDataset
from genpu.models.stage2 import Stage2Model

# ── Default config ──────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data
    "data_dir": "/scratch/gpfs/IOJALVO/genpu_data/preprocessed",
    "n_shards": 10,         # number of shards to use (None = all)
    "val_fraction": 0.05,

    # Model
    "embed_dim": 128,
    "n_denoiser_layers": 4,
    "n_heads": 4,
    "max_tracker_hits": 32,
    "max_calo_hits": 64,
    "diffusion_timesteps": 1000,
    "dropout": 0.0,

    # Training
    "batch_size": 1024,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_steps": 1000,
    "max_steps": 100_000,
    "log_every": 100,
    "val_every": 2000,
    "save_every": 10_000,
    "num_workers": 4,

    # Loss weights
    "count_loss_weight": 1.0,
    "tracker_diff_weight": 1.0,
    "calo_diff_weight": 1.0,

    # Output
    "output_dir": "/scratch/gpfs/IOJALVO/genpu_data/checkpoints/stage2",
    "run_name": None,
}


def get_lr_schedule(optimizer, warmup_steps, max_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    # Allow overriding individual config values
    for key, val in DEFAULT_CONFIG.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda x: x.lower() == "true", default=None)
        elif isinstance(val, int):
            parser.add_argument(f"--{key}", type=int, default=None)
        elif isinstance(val, float):
            parser.add_argument(f"--{key}", type=float, default=None)
        elif isinstance(val, str) or val is None:
            parser.add_argument(f"--{key}", type=str, default=None)
    args = parser.parse_args()

    # Build config
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    # Override from CLI
    for key in DEFAULT_CONFIG:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val

    if cfg["run_name"] is None:
        cfg["run_name"] = f"stage2_{time.strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────

    shard_indices = list(range(cfg["n_shards"])) if cfg["n_shards"] else None
    print(f"Loading dataset from {cfg['data_dir']}...")
    dataset = ParticleHitDataset(
        cfg["data_dir"],
        shard_indices=shard_indices,
        max_tracker_hits=cfg["max_tracker_hits"],
        max_calo_hits=cfg["max_calo_hits"],
    )
    print(f"  {len(dataset)} particles loaded")

    # Train/val split
    n_val = int(len(dataset) * cfg["val_fraction"])
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=ParticleHitDataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=ParticleHitDataset.collate_fn,
        pin_memory=True,
    )

    print(f"  Train: {n_train}, Val: {n_val}")

    # ── Model ───────────────────────────────────────────────────────────────

    model = Stage2Model(
        embed_dim=cfg["embed_dim"],
        max_tracker_hits=cfg["max_tracker_hits"],
        max_calo_hits=cfg["max_calo_hits"],
        n_denoiser_layers=cfg["n_denoiser_layers"],
        n_heads=cfg["n_heads"],
        diffusion_timesteps=cfg["diffusion_timesteps"],
        dropout=cfg["dropout"],
    ).to(device)

    n_params = count_parameters(model)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = get_lr_schedule(optimizer, cfg["warmup_steps"], cfg["max_steps"])

    # ── Logging ─────────────────────────────────────────────────────────────

    try:
        import wandb
        wandb.init(project="genpu", name=cfg["run_name"], config=cfg)
        use_wandb = True
        print("  Logging to wandb")
    except ImportError:
        use_wandb = False
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output_dir / "tb")
        print(f"  Logging to tensorboard: {output_dir / 'tb'}")

    def log_metrics(metrics, step):
        if use_wandb:
            wandb.log(metrics, step=step)
        else:
            for k, v in metrics.items():
                writer.add_scalar(k, v, step)

    # ── Training loop ───────────────────────────────────────────────────────

    model.train()
    step = 0
    epoch = 0
    running_losses = {}
    t0 = time.time()

    print(f"\nTraining for {cfg['max_steps']} steps...")

    while step < cfg["max_steps"]:
        epoch += 1
        for batch in train_loader:
            if step >= cfg["max_steps"]:
                break

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward + backward
            losses = model.training_step(batch)

            # Apply loss weights
            weighted_loss = (
                cfg["count_loss_weight"] * losses["total_count_loss"]
                + cfg["tracker_diff_weight"] * losses["tracker_diffusion_loss"]
                + cfg["calo_diff_weight"] * losses["calo_diffusion_loss"]
            )

            optimizer.zero_grad()
            weighted_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track running losses
            for k, v in losses.items():
                if k not in running_losses:
                    running_losses[k] = 0.0
                running_losses[k] += v.item()

            step += 1

            # Log
            if step % cfg["log_every"] == 0:
                avg_losses = {f"train/{k}": v / cfg["log_every"] for k, v in running_losses.items()}
                avg_losses["train/lr"] = scheduler.get_last_lr()[0]
                avg_losses["train/epoch"] = epoch
                elapsed = time.time() - t0
                avg_losses["train/steps_per_sec"] = step / elapsed

                log_metrics(avg_losses, step)
                print(
                    f"  step {step:>6d} | "
                    f"loss {avg_losses['train/total_loss']:.4f} | "
                    f"count {avg_losses['train/total_count_loss']:.4f} | "
                    f"trk_diff {avg_losses['train/tracker_diffusion_loss']:.4f} | "
                    f"cal_diff {avg_losses['train/calo_diffusion_loss']:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )
                running_losses = {}

            # Validate
            if step % cfg["val_every"] == 0:
                model.eval()
                val_losses = {}
                n_val_batches = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                        vl = model.training_step(val_batch)
                        for k, v in vl.items():
                            val_losses[k] = val_losses.get(k, 0.0) + v.item()
                        n_val_batches += 1
                        if n_val_batches >= 50:  # cap validation
                            break

                avg_val = {f"val/{k}": v / n_val_batches for k, v in val_losses.items()}
                log_metrics(avg_val, step)
                print(
                    f"  [VAL] step {step:>6d} | "
                    f"loss {avg_val['val/total_loss']:.4f} | "
                    f"count {avg_val['val/total_count_loss']:.4f} | "
                    f"trk_diff {avg_val['val/tracker_diffusion_loss']:.4f} | "
                    f"cal_diff {avg_val['val/calo_diffusion_loss']:.4f}"
                )
                model.train()

            # Save checkpoint
            if step % cfg["save_every"] == 0:
                ckpt_path = output_dir / f"checkpoint_{step:06d}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    ckpt_path = output_dir / "checkpoint_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }, ckpt_path)
    print(f"\nTraining complete. Final checkpoint: {ckpt_path}")

    if use_wandb:
        wandb.finish()
    else:
        writer.close()


if __name__ == "__main__":
    main()
