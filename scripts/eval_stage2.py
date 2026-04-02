#!/usr/bin/env python
"""Evaluate Stage 2 model: generate hits for held-out particles, produce validation plots.

Usage:
    python scripts/eval_stage2.py --checkpoint PATH [--n_samples 5000] [--n_steps 50]

Produces plots in {checkpoint_dir}/eval/:
1. Hit count distributions (predicted vs true)
2. Tracker hit spatial distributions (r, phi, z per layer)
3. Calo hit distributions (eta, phi, energy per detector)
4. Per-particle hit count accuracy by PDG class and pT bin
5. Example event displays
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.dataset import ParticleHitDataset
from genpu.models.stage2 import Stage2Model
from genpu.preprocessing import N_PDG_CLASSES, PDG_TO_CLASS

# Reverse map for plot labels
CLASS_TO_NAME = {
    0: "e-", 1: "e+", 2: "γ", 3: "π+", 4: "π-",
    5: "K+", 6: "K-", 7: "p", 8: "p̄",
    9: "n", 10: "n̄", 11: "μ-", 12: "μ+",
    13: "K0L", 14: "K0S", 15: "π0", 16: "other",
}


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Stage2Model(
        embed_dim=cfg["embed_dim"],
        max_tracker_hits=cfg["max_tracker_hits"],
        max_calo_hits=cfg["max_calo_hits"],
        n_denoiser_layers=cfg["n_denoiser_layers"],
        n_heads=cfg["n_heads"],
        diffusion_timesteps=cfg["diffusion_timesteps"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/scratch/gpfs/IOJALVO/genpu_data/preprocessed")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of particles to evaluate")
    parser.add_argument("--n_steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--eval_shards", type=int, nargs="+", default=[9], help="Shard indices for eval (should not overlap with training)")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output dir
    ckpt_path = Path(args.checkpoint)
    eval_dir = ckpt_path.parent / "eval"
    eval_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from {ckpt_path}...")
    model, cfg = load_model(ckpt_path, device)

    # Load eval data
    print(f"Loading eval data (shards {args.eval_shards})...")
    dataset = ParticleHitDataset(
        args.data_dir,
        shard_indices=args.eval_shards,
        max_tracker_hits=cfg["max_tracker_hits"],
        max_calo_hits=cfg["max_calo_hits"],
    )

    n_eval = min(args.n_samples, len(dataset))
    print(f"  Evaluating {n_eval} particles")

    # Collect ground truth and generated hits
    true_n_trk, true_n_cal = [], []
    gen_n_trk, gen_n_cal = [], []
    true_trk_features, gen_trk_features = [], []
    true_cal_features, gen_cal_features = [], []
    particle_pdg_classes = []
    particle_log_pt = []

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(n_eval)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ParticleHitDataset.collate_fn,
        num_workers=2,
    )

    for batch in tqdm(loader, desc="Generating"):
        pf = batch["particle_features"].to(device)
        B = pf.shape[0]

        # Ground truth
        true_n_trk.append(batch["n_tracker"].numpy())
        true_n_cal.append(batch["n_calo"].numpy())
        particle_pdg_classes.append(pf[:, 3].cpu().numpy())
        particle_log_pt.append(pf[:, 0].cpu().numpy())

        # Collect true hit features (unpadded)
        for i in range(B):
            nt = batch["n_tracker"][i].item()
            nc = batch["n_calo"][i].item()
            if nt > 0:
                true_trk_features.append(batch["tracker_hits"][i, :nt].numpy())
            if nc > 0:
                true_cal_features.append(batch["calo_hits"][i, :nc].numpy())

        # Generate
        gen = model.generate(pf, n_steps=args.n_steps)
        gen_n_trk.append(gen["n_tracker"].cpu().numpy())
        gen_n_cal.append(gen["n_calo"].cpu().numpy())

        for i in range(B):
            nt = gen["n_tracker"][i].item()
            nc = gen["n_calo"][i].item()
            if nt > 0:
                gen_trk_features.append(gen["tracker_hits"][i, :nt].cpu().numpy())
            if nc > 0:
                gen_cal_features.append(gen["calo_hits"][i, :nc].cpu().numpy())

    true_n_trk = np.concatenate(true_n_trk)
    true_n_cal = np.concatenate(true_n_cal)
    gen_n_trk = np.concatenate(gen_n_trk)
    gen_n_cal = np.concatenate(gen_n_cal)
    particle_pdg_classes = np.concatenate(particle_pdg_classes).astype(int)
    particle_log_pt = np.concatenate(particle_log_pt)

    # ── Plot 1: Hit count distributions ─────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_trk = max(true_n_trk.max(), gen_n_trk.max()) + 1
    bins = np.arange(0, min(max_trk, 40))
    axes[0].hist(true_n_trk, bins=bins, alpha=0.6, label="True", density=True)
    axes[0].hist(gen_n_trk, bins=bins, alpha=0.6, label="Generated", density=True)
    axes[0].set_xlabel("Tracker hits per particle")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Tracker hit count distribution")
    axes[0].legend()

    max_cal = max(true_n_cal.max(), gen_n_cal.max()) + 1
    bins = np.arange(0, min(max_cal, 80))
    axes[1].hist(true_n_cal, bins=bins, alpha=0.6, label="True", density=True)
    axes[1].hist(gen_n_cal, bins=bins, alpha=0.6, label="Generated", density=True)
    axes[1].set_xlabel("Calo hits per particle")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Calo hit count distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(eval_dir / "hit_count_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Hit counts by PDG class ─────────────────────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    classes_present = sorted(set(particle_pdg_classes))
    class_names = [CLASS_TO_NAME.get(c, str(c)) for c in classes_present]

    true_trk_means = [true_n_trk[particle_pdg_classes == c].mean() for c in classes_present]
    gen_trk_means = [gen_n_trk[particle_pdg_classes == c].mean() for c in classes_present]

    x = np.arange(len(classes_present))
    w = 0.35
    axes[0].bar(x - w / 2, true_trk_means, w, label="True", alpha=0.7)
    axes[0].bar(x + w / 2, gen_trk_means, w, label="Generated", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names)
    axes[0].set_ylabel("Mean tracker hits")
    axes[0].set_title("Mean tracker hits by particle species")
    axes[0].legend()

    true_cal_means = [true_n_cal[particle_pdg_classes == c].mean() for c in classes_present]
    gen_cal_means = [gen_n_cal[particle_pdg_classes == c].mean() for c in classes_present]

    axes[1].bar(x - w / 2, true_cal_means, w, label="True", alpha=0.7)
    axes[1].bar(x + w / 2, gen_cal_means, w, label="Generated", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylabel("Mean calo hits")
    axes[1].set_title("Mean calo hits by particle species")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(eval_dir / "hit_counts_by_pdg.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Hit counts vs pT ────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pt_bins = np.linspace(particle_log_pt.min(), particle_log_pt.max(), 20)
    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    pt_idx = np.digitize(particle_log_pt, pt_bins) - 1

    true_trk_vs_pt = [true_n_trk[pt_idx == i].mean() if (pt_idx == i).any() else 0 for i in range(len(pt_centers))]
    gen_trk_vs_pt = [gen_n_trk[pt_idx == i].mean() if (pt_idx == i).any() else 0 for i in range(len(pt_centers))]

    axes[0].plot(np.exp(pt_centers), true_trk_vs_pt, "o-", label="True", alpha=0.7)
    axes[0].plot(np.exp(pt_centers), gen_trk_vs_pt, "s--", label="Generated", alpha=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("pT [GeV]")
    axes[0].set_ylabel("Mean tracker hits")
    axes[0].set_title("Mean tracker hits vs pT")
    axes[0].legend()

    true_cal_vs_pt = [true_n_cal[pt_idx == i].mean() if (pt_idx == i).any() else 0 for i in range(len(pt_centers))]
    gen_cal_vs_pt = [gen_n_cal[pt_idx == i].mean() if (pt_idx == i).any() else 0 for i in range(len(pt_centers))]

    axes[1].plot(np.exp(pt_centers), true_cal_vs_pt, "o-", label="True", alpha=0.7)
    axes[1].plot(np.exp(pt_centers), gen_cal_vs_pt, "s--", label="Generated", alpha=0.7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("pT [GeV]")
    axes[1].set_ylabel("Mean calo hits")
    axes[1].set_title("Mean calo hits vs pT")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(eval_dir / "hit_counts_vs_pt.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Tracker hit spatial distributions ───────────────────────────

    if true_trk_features and gen_trk_features:
        true_trk = np.concatenate(true_trk_features)
        gen_trk = np.concatenate(gen_trk_features)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        trk_names = ["r [mm]", "φ", "z [mm]", "time", "volume_id", "layer_id"]

        for i, name in enumerate(trk_names):
            ax = axes[i // 3, i % 3]
            lo = min(true_trk[:, i].min(), gen_trk[:, i].min())
            hi = max(true_trk[:, i].max(), gen_trk[:, i].max())
            bins = np.linspace(lo, hi, 80)
            ax.hist(true_trk[:, i], bins=bins, alpha=0.6, label="True", density=True)
            ax.hist(gen_trk[:, i], bins=bins, alpha=0.6, label="Generated", density=True)
            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            ax.legend()

        plt.suptitle("Tracker hit feature distributions", fontsize=14)
        plt.tight_layout()
        plt.savefig(eval_dir / "tracker_hit_features.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Plot 5: Calo hit distributions ──────────────────────────────────────

    if true_cal_features and gen_cal_features:
        true_cal = np.concatenate(true_cal_features)
        gen_cal = np.concatenate(gen_cal_features)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        cal_names = ["η", "φ", "log(energy)", "contrib_frac", "detector"]

        for i, name in enumerate(cal_names):
            ax = axes[i]
            lo = min(true_cal[:, i].min(), gen_cal[:, i].min())
            hi = max(true_cal[:, i].max(), gen_cal[:, i].max())
            bins = np.linspace(lo, hi, 80)
            ax.hist(true_cal[:, i], bins=bins, alpha=0.6, label="True", density=True)
            ax.hist(gen_cal[:, i], bins=bins, alpha=0.6, label="Generated", density=True)
            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            ax.legend()

        plt.suptitle("Calo hit feature distributions", fontsize=14)
        plt.tight_layout()
        plt.savefig(eval_dir / "calo_hit_features.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Plot 6: Example particles ───────────────────────────────────────────

    # Pick a few interesting particles (high pT charged)
    charged_mask = np.isin(particle_pdg_classes, [0, 1, 3, 4, 7, 8, 11, 12])
    high_pt_mask = particle_log_pt > 0  # pT > 1 GeV
    interesting = np.where(charged_mask & high_pt_mask)[0]

    if len(interesting) > 0:
        n_show = min(6, len(interesting))
        show_idx = interesting[:n_show]

        fig, axes = plt.subplots(n_show, 2, figsize=(14, 4 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        for row, idx in enumerate(show_idx):
            pf = dataset[idx]
            pdg_cls = int(pf["particle_features"][3].item())
            pt = np.exp(pf["particle_features"][0].item())

            # True hits
            nt = pf["n_tracker"]
            if nt > 0:
                trk = pf["tracker_hits"][:nt].numpy()
                axes[row, 0].scatter(trk[:, 2], trk[:, 0], s=20, c="blue", label="True")

            # Generate
            pf_tensor = pf["particle_features"].unsqueeze(0).to(device)
            gen = model.generate(pf_tensor, n_steps=args.n_steps)
            gnt = gen["n_tracker"][0].item()
            if gnt > 0:
                gtrk = gen["tracker_hits"][0, :gnt].cpu().numpy()
                axes[row, 0].scatter(gtrk[:, 2], gtrk[:, 0], s=20, c="red", marker="x", label="Generated")

            axes[row, 0].set_xlabel("z [mm]")
            axes[row, 0].set_ylabel("r [mm]")
            axes[row, 0].set_title(f"{CLASS_TO_NAME[pdg_cls]}, pT={pt:.2f} GeV, n_trk: true={nt} gen={gnt}")
            axes[row, 0].legend()

            # Calo
            nc = pf["n_calo"]
            if nc > 0:
                cal = pf["calo_hits"][:nc].numpy()
                axes[row, 1].scatter(cal[:, 0], cal[:, 1], s=20, c="blue", label="True")

            gnc = gen["n_calo"][0].item()
            if gnc > 0:
                gcal = gen["calo_hits"][0, :gnc].cpu().numpy()
                axes[row, 1].scatter(gcal[:, 0], gcal[:, 1], s=20, c="red", marker="x", label="Generated")

            axes[row, 1].set_xlabel("η")
            axes[row, 1].set_ylabel("φ")
            axes[row, 1].set_title(f"{CLASS_TO_NAME[pdg_cls]}, pT={pt:.2f} GeV, n_cal: true={nc} gen={gnc}")
            axes[row, 1].legend()

        plt.tight_layout()
        plt.savefig(eval_dir / "example_particles.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Summary stats ───────────────────────────────────────────────────────

    summary = {
        "n_particles_evaluated": int(n_eval),
        "tracker_count_accuracy": {
            "exact_match": float((true_n_trk == gen_n_trk).mean()),
            "within_1": float((np.abs(true_n_trk - gen_n_trk) <= 1).mean()),
            "within_3": float((np.abs(true_n_trk - gen_n_trk) <= 3).mean()),
            "mean_true": float(true_n_trk.mean()),
            "mean_gen": float(gen_n_trk.mean()),
        },
        "calo_count_accuracy": {
            "exact_match": float((true_n_cal == gen_n_cal).mean()),
            "within_1": float((np.abs(true_n_cal - gen_n_cal) <= 1).mean()),
            "within_3": float((np.abs(true_n_cal - gen_n_cal) <= 3).mean()),
            "mean_true": float(true_n_cal.mean()),
            "mean_gen": float(gen_n_cal.mean()),
        },
    }

    with open(eval_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Particles evaluated: {n_eval}")
    print(f"\nTracker hit counts:")
    print(f"  Exact match:  {summary['tracker_count_accuracy']['exact_match']:.1%}")
    print(f"  Within ±1:    {summary['tracker_count_accuracy']['within_1']:.1%}")
    print(f"  Within ±3:    {summary['tracker_count_accuracy']['within_3']:.1%}")
    print(f"  Mean true:    {summary['tracker_count_accuracy']['mean_true']:.2f}")
    print(f"  Mean gen:     {summary['tracker_count_accuracy']['mean_gen']:.2f}")
    print(f"\nCalo hit counts:")
    print(f"  Exact match:  {summary['calo_count_accuracy']['exact_match']:.1%}")
    print(f"  Within ±1:    {summary['calo_count_accuracy']['within_1']:.1%}")
    print(f"  Within ±3:    {summary['calo_count_accuracy']['within_3']:.1%}")
    print(f"  Mean true:    {summary['calo_count_accuracy']['mean_true']:.2f}")
    print(f"  Mean gen:     {summary['calo_count_accuracy']['mean_gen']:.2f}")
    print(f"\nPlots saved to {eval_dir}/")


if __name__ == "__main__":
    main()
