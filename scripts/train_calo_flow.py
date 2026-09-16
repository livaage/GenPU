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
    ap.add_argument("--max_particles", type=int, default=0, help="0=all; subsample showers (smoke)")
    ap.add_argument("--pt_batch", type=int, default=16384)
    ap.add_argument("--glob_batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--glob_weight", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--run_name", default="calo_flow_photon_v1")
    ap.add_argument("--pos_transform", choices=["none", "arcsinh", "quantile"], default="none",
                    help="reshape peaked marginals so the flow/mixture fit a Gaussian "
                         "(quantile = bounded normal-quantile normalisation; recommended)")
    ap.add_argument("--no_energy_glob", action="store_true",
                    help="pre-audit behaviour: condition the energy head on particle features only "
                         "(measured R^2 0.05 on the per-shower energy scale vs 0.85 with the global)")
    ap.add_argument("--no_floor_atom", action="store_true",
                    help="drop the at-floor Bernoulli point mass and train the energy mixture on "
                         "ALL cells. Measured 2026-09-07 on 13,054,567 held-out e± cells: ZERO sit "
                         "at log(5e-5), 0.387%% sit below it, and the threshold is a x19 DENSITY "
                         "STEP -- so the atom is a point mass on a value the data never takes, and "
                         "the mixture beside it is trained with a hole the atom fills back in. "
                         "Open risk: a Gaussian mixture cannot represent a discontinuity and may "
                         "smear the step; if it does, the fix is to TRUNCATE the mixture at the "
                         "threshold, not to restore the atom.")
    ap.add_argument("--separate_trunks", action="store_true",
                    help="give each head its own ParticleConditioning MLP. With the shared trunk, "
                         "changing one head's task moves what the others see (measured twice); this "
                         "makes that impossible for ~+11k params (~4%%).")
    ap.add_argument("--ctx_norm", action="store_true",
                    help="context-normalise the shower CORE (glob dims 2,3) per (charge sign x pT "
                         "bin) before the pooled quantile transform — fixes the under-dispersed "
                         "per-bin core spread (0.58x real for e±, 0.94x for pions)")
    ap.add_argument("--ctx_pt_bins", type=int, default=8)
    ap.add_argument("--energy_glob_idx", type=int, nargs="+", default=None,
                    help="which standardised globals the ENERGY head sees, by index "
                         "(0=total_logE, 1=log_n). `--energy_glob_idx 1` gives it MULTIPLICITY "
                         "without the energy scale: frac_near_floor is a function of multiplicity, "
                         "which is the GlobalHead's variable, and with --no_energy_glob the energy "
                         "head can only infer it implicitly through a shared trunk — five separate "
                         "changes that disturbed that route all degraded frac_near_floor. Distinct "
                         "from the ruled-out full-global conditioning, which failed on dim 0.")
    ap.add_argument("--floor_n_buckets", type=int, default=0,
                    help="route the energy head's at-floor Bernoulli through its own net, which also "
                         "sees an EMBEDDING of the shower's cell count (n clamped to [1,N]); 0 = off. "
                         "The Gaussian mixture is untouched. Targets the 2026-08-16 measurement of "
                         "p(floor|n): the e± error is ENTIRELY at n=1 (real 0.011 vs generated 0.061, "
                         "a 6x excess) and gone by n>=9, while the pion is missing a monotone SLOPE "
                         "(real 0.018->0.032, generated flat). A single log_n scalar cannot express "
                         "either — hence --energy_glob_idx 1 being worth only 0.618->0.574. Unlike "
                         "`partition` (which forced the same n=1 fix at generation time and detonated "
                         "the marginal) this changes only a discrete membership decision, never a "
                         "cell's energy. 32 is a sensible N: exact for n=1..31, lumps the rest.")
    ap.add_argument("--energy_pos", action="store_true",
                    help="condition the ENERGY head on each cell's own position: standardised "
                         "depth-within-section + transverse radius in units of the shower width, "
                         "plus the 4-way ECAL/HCAL x barrel/endcap region token. Until now every "
                         "input to that head was a per-SHOWER quantity, so cell energy was "
                         "independent of cell position by construction; measured within-shower "
                         "rho(logE, depth) real vs generated is p +0.52/0.000, pi +0.20/0.000, "
                         "gamma -0.11/0.000 (jobs 13032162, 13033802). Needs a v2 slice carrying "
                         "point_depth_local / point_region.")
    ap.add_argument("--anchor_cond", action="store_true",
                    help="condition the GlobalHead (only) on the anchor: [a_eta,a_phi,|a|] + a "
                         "branch one-hot. Fixes the measured branch blindness — the real residual "
                         "core scale differs 10x (pion) / 15x (e±) between face-reaching showers "
                         "and curlers, which the trunk's smooth features cannot express.")
    ap.add_argument("--qt_total", action="store_true",
                    help="quantile-normalise total_logE (glob dim 0) too, so the mixture's inverse is "
                         "BOUNDED — an unbounded draw put the photon total at 1e9 x E_true")
    ap.add_argument("--count_dither", action="store_true",
                    help="dequantize the discrete count: log_n <- log(n + U(-0.5,0.5)) so the "
                         "continuous mixture can fit the n=1 atom (recovered by round at gen)")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--joint_energy", action="store_true",
                    help="A-prime (2026-09-16): the point flow emits log-E as a 4th coordinate, so "
                         "position and energy are generated jointly and EnergyHead is not trained. "
                         "Sub-threshold cells are DROPPED at generation (zero-suppression), never "
                         "clamped -- clamping is what made the original energy-in-flow model pile "
                         "cells at 50 keV.")
    ap.add_argument("--point_attn", type=int, default=0,
                    help="set-attention layers in the point flow: cells of a shower attend to each "
                         "other. 0 = the historical per-cell MLP (cells i.i.d. given conditioning).")
    ap.add_argument("--attn_dim", type=int, default=128)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--attn_self_only", action="store_true",
                    help="CONTROL: identical set-attention network, attention masked to the "
                         "diagonal so each cell sees only itself. Isolates 'cells see each other' "
                         "from the architecture/capacity change.")
    ap.add_argument("--set_budget", type=int, default=24576,
                    help="padded tokens per shower block (training batches and sampling)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch seed (init + batch sampling). Every logged run before 2026-08-16 used "
                         "0, so keep it there for like-for-like comparisons; change it only to get a "
                         "REPLICATE, which is what tells a real gate delta from seed noise.")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std", "glob_mean", "glob_std", "pts_mean", "pts_std"]}
    cont, pdg, glob, pts, off = d["cont"], d["pdg"], d["glob"], d["points_flat"], d["offsets"]
    S = cont.shape[0]
    if args.max_particles and S > args.max_particles:  # smoke-test subsample (contiguous showers)
        K = args.max_particles
        cont, pdg, glob = cont[:K], pdg[:K], glob[:K]
        off = off[:K + 1]; pts = pts[:off[-1]]; S = K

    # deterministic shower-level train/val split
    rng = np.random.default_rng(0)
    perm = rng.permutation(S)
    n_val = int(S * args.val_frac)
    val_sh = np.zeros(S, bool); val_sh[perm[:n_val]] = True
    tr_sh = ~val_sh

    npt = np.diff(off)
    point_shower = np.repeat(np.arange(S), npt)          # (P,) shower id per point
    tr_pt = tr_sh[point_shower]

    log_floor = float(d["log_floor"]) if "log_floor" in d else float(np.log(5e-5))

    # dequantize the count: log_n = log(n) is discrete with a big atom at n=1 (~23%), which the
    # continuous mixture smooths -> gen under-produces single-cell showers (and thus the width=0
    # / lead_frac=1 / logE_std=0 spikes). Spreading each integer into a continuous block lets the
    # mixture fit it; round(exp(.)) at generation recovers the integer.
    if args.count_dither:
        glob = glob.copy()
        n = np.round(np.exp(glob[:, 1])).astype(np.float32)
        glob[:, 1] = np.log(np.clip(n + rng.uniform(-0.5, 0.5, len(n)).astype(np.float32), 0.5, None))
        norm = dict(norm)
        gm = norm["glob_mean"].copy(); gs = norm["glob_std"].copy()
        gm[1] = glob[:, 1].mean(); gs[1] = glob[:, 1].std()
        norm["glob_mean"] = gm.astype(np.float32); norm["glob_std"] = gs.astype(np.float32)
        print(f"count dither: log_n mean {gm[1]:.3f} std {gs[1]:.3f}")

    # position coordinate warp: arcsinh(delta/s) de-peaks the near-delta d_eta/d_phi so
    # the flow fits a broad distribution; s per-dim = median(|delta|). pts_mean/std are
    # then recomputed over WARP space (the flow standardizes in warp space; unstd_pos
    # inverts with sinh). No slice rebuild needed.
    G = len(norm["glob_mean"])
    # CONTEXT normalisation of the core dims: per (charge sign x pT bin), from truth conditioning.
    # Measured 2026-08-13: the pooled quantile transform matches the core marginal but leaves the
    # per-bin scale varying 2.4x (e±) / 4.0x (pion), and the mixture then generates only 0.58x
    # (e±) / 0.94x (pion) of the real per-bin spread. Normalising each context to zero-median /
    # unit-IQR first makes the pooled transform act on a representative shape.
    ctx_kwargs = {}
    if args.ctx_norm and G >= 4:
        n_pt = args.ctx_pt_bins
        edges = np.quantile(cont[:, 0], np.linspace(0, 1, n_pt + 1))[1:-1].astype(np.float32)
        pt_bin = np.digitize(cont[:, 0], edges)
        q_idx = np.sign(cont[:, 3]).astype(int) + 1                  # 0 neg, 1 neutral, 2 pos
        ctx = np.clip(q_idx * n_pt + pt_bin, 0, 3 * n_pt - 1)
        C = 3 * n_pt
        ctx_loc = np.zeros((C, G), np.float32); ctx_scale = np.ones((C, G), np.float32)
        ctx_mask = np.zeros(G, np.float32); ctx_mask[2] = ctx_mask[3] = 1.0   # core dims only
        for c in range(C):
            m = ctx == c
            if m.sum() < 200:            # too few to fit: leave identity, the pooled path handles it
                continue
            for j in (2, 3):
                q1, q3 = np.percentile(glob[m, j], [25, 75])
                ctx_loc[c, j] = np.median(glob[m, j])
                ctx_scale[c, j] = max((q3 - q1) / 1.349, 1e-6)
        # the quantile grids below must be fit on the CONTEXT-NORMALISED values
        glob = glob.copy()
        for j in (2, 3):
            glob[:, j] = (glob[:, j] - ctx_loc[ctx, j]) / ctx_scale[ctx, j]
        ctx_kwargs = dict(ctx_pt_edges=edges, ctx_loc=ctx_loc, ctx_scale=ctx_scale, ctx_mask=ctx_mask)
        sc = ctx_scale[:, 3][ctx_scale[:, 3] != 1.0]
        print(f"ctx_norm ON: {C} contexts (3 charge x {n_pt} pT bins), core dims [2,3]; "
              f"core_phi per-context scale spans {sc.min():.3f}..{sc.max():.3f} "
              f"({sc.max()/max(sc.min(),1e-9):.1f}x) -> normalised to ~1")

    qt_kwargs = {}
    if args.pos_transform == "quantile":
        from scipy.special import ndtri  # inverse standard-normal CDF
        K = 256
        probs = np.linspace(0.5 / K, 1 - 0.5 / K, K)                 # avoid 0/1 -> +-inf
        qt_z = ndtri(probs).astype(np.float32)                       # Gaussian quantile grid (monotone)
        qt_pos_x = np.quantile(pts[:, :2], probs, axis=0).astype(np.float32)   # (K,2) monotone per col
        # quantile-normalise the CORE dims (2,3); other globals keep z-score (ramp = unused).
        qt_glob_x = np.tile(np.linspace(-1, 1, K, dtype=np.float32)[:, None], (1, G))
        qt_glob_mask = np.zeros(G, np.float32)
        for j in range(G):
            # core dims (2,3) and log_width (4) are sharply peaked / heavy-tailed -> quantile them
            # (their inverse is then BOUNDED to the data's own range); total_logE only on request
            if (G >= 4 and j in (2, 3)) or (G >= 5 and j == 4) or (args.qt_total and j == 0):
                qt_glob_x[:, j] = np.quantile(glob[:, j], probs); qt_glob_mask[j] = 1.0
        qt_kwargs = dict(qt_z=qt_z, qt_pos_x=qt_pos_x, qt_glob_x=qt_glob_x, qt_glob_mask=qt_glob_mask)
        print(f"quantile normalise: pos dims [0,1], glob dims {np.where(qt_glob_mask>0)[0].tolist()}  K={K}")
    # sampling bounds for the energy head: the hardest cell seen in training, overall and PER
    # SPECIES (a multi-species slice's overall max comes from hadrons and is ~4x too loose for EM)
    # pos_dim from the slice: (P,3)=v1 2D, (P,4)=v2 with depth. Energy is always the LAST column,
    # so index -1 rather than a hard 2 -- that assumption is what would silently train on depth.
    pos_dim = pts.shape[1] - 1
    print(f"points_flat {pts.shape} -> pos_dim {pos_dim}"
          f"{'  (v2: includes longitudinal depth)' if pos_dim == 3 else ''}")
    logE_max = float(pts[:, -1].max())
    from genpu.preprocessing import N_PDG_CLASSES
    from genpu.flow.calo_flow import POS_FEAT_NAMES
    from genpu.calo_geom import N_CALO_REGIONS
    pt_pdg_np = np.repeat(pdg.astype(np.int64), npt)
    logE_max_pdg = np.full(N_PDG_CLASSES, np.inf, np.float32)
    for c in np.unique(pt_pdg_np):
        logE_max_pdg[int(c)] = pts[pt_pdg_np == c, -1].max()   # LAST column = energy; index 2 is DEPTH under v2
    # PER-SPECIES CELL-COUNT BOUND, same contract as logE_max_pdg: the largest shower this class
    # actually produced. `sample_showers` used a bare max_cells=128 literal until 2026-09-14, which
    # on this slice truncates 2.10% of showers / 6.87% of all cells, almost all hadronic
    # (pbar 25.3% of showers, mu± 0.00%) -- so one shared bound is as wrong here as it was there.
    _pdg_sh = pdg.astype(np.int64)
    n_max_pdg = np.full(N_PDG_CLASSES, np.inf, np.float32)
    for c in np.unique(_pdg_sh):
        n_max_pdg[int(c)] = float(npt[_pdg_sh == c].max())
    print("  per-class n_max: " + ", ".join(
        f"{int(c)}:{n_max_pdg[int(c)]:.0f}" for c in np.unique(_pdg_sh)))
    # PER-CELL POSITION FEATURES (2026-08-27). Read from the slice's section fields, never
    # recomputed here -- `build_calo_slice_v2` is the one implementation, the same rule that keeps
    # the anchor from drifting between training and generation.
    pos_feat_np = region_np = None
    if args.energy_pos:
        need = {"point_depth_local", "point_region"}
        if not need <= set(d.files):
            raise SystemExit(f"--energy_pos needs {sorted(need)} in the slice; rebuild with "
                             f"build_calo_slice_v2.py (2026-08-27 or later)")
        dloc = d["point_depth_local"].astype(np.float32)[:pts.shape[0]]
        region_np = d["point_region"].astype(np.int64)[:pts.shape[0]]
        _w = np.sqrt(np.maximum(
            np.bincount(point_shower, weights=(pts[:, 0] ** 2 + pts[:, 1] ** 2), minlength=S) / npt, 1e-12))
        r_over_w = (np.hypot(pts[:, 0], pts[:, 1]) / np.maximum(_w[point_shower], 1e-6)).astype(np.float32)
        pos_feat_np = np.stack([dloc, r_over_w], 1)
        # standardise on the TRAIN split only, so the val split cannot leak into the norm
        _m = pos_feat_np[tr_pt].mean(0); _s = pos_feat_np[tr_pt].std(0) + 1e-6
        print(f"--energy_pos ON: features {POS_FEAT_NAMES} mean {_m.round(3).tolist()} "
              f"std {_s.round(3).tolist()}; region mix "
              f"{ {int(k): round(float(v)/len(region_np), 4) for k, v in zip(*np.unique(region_np, return_counts=True))} }")
    width_norm = bool(int(d["width_normalized"])) if "width_normalized" in d else False
    # Phase 1: the slice decides the core frame; the model just records it so the generation path
    # knows it must be handed the anchor.
    core_anchored = str(d["core_anchor"]) if "core_anchor" in d else "none"
    # anchor conditioning features are standardised on this slice; the stats ride in buffers so
    # generation builds the identical vector (CaloFlow.anchor_feats is the single implementation).
    anchor_kwargs = {}
    if args.anchor_cond:
        if core_anchored == "none":
            raise SystemExit("--anchor_cond needs a slice built with --core_anchor")
        anc_np = d["anchor"]
        a3 = np.concatenate([anc_np, np.hypot(anc_np[:, 0], anc_np[:, 1])[:, None]], 1)
        anchor_kwargs = dict(anchor_cond=True, anchor_mean=a3.mean(0), anchor_std=a3.std(0) + 1e-6)
    je_mean = je_std = None
    if args.joint_energy:
        _le = pts[:, -1][tr_pt]                     # LAST column = energy, train split only
        je_mean, je_std = float(_le.mean()), float(_le.std() + 1e-6)
        print(f"--joint_energy ON: log-E standardised with mean {je_mean:.3f} std {je_std:.3f}; "
              f"EnergyHead NOT trained")
    if args.point_attn:
        print(f"--point_attn {args.point_attn} (dim {args.attn_dim}, heads {args.attn_heads}, "
              f"budget {args.set_budget}){'  ** SELF-ONLY CONTROL **' if args.attn_self_only else ''}")
    set_kwargs = dict(joint_energy=args.joint_energy, point_attn=args.point_attn,
                      attn_dim=args.attn_dim, attn_heads=args.attn_heads,
                      attn_self_only=args.attn_self_only, set_budget=args.set_budget)
    if je_mean is not None:
        set_kwargs.update(joint_e_mean=je_mean, joint_e_std=je_std)
    model = CaloFlow(norm, log_floor=log_floor, energy_use_glob=not args.no_energy_glob,
                     logE_max=logE_max, logE_max_pdg=logE_max_pdg, n_max_pdg=n_max_pdg,
                     width_norm=width_norm,
                     separate_trunks=args.separate_trunks,
                     core_anchored=core_anchored != "none", energy_glob_idx=args.energy_glob_idx,
                     floor_n_buckets=args.floor_n_buckets, pos_dim=pos_dim,
                     floor_atom=not args.no_floor_atom,
                     energy_pos_dim=(pos_feat_np.shape[1] if pos_feat_np is not None else 0),
                     energy_n_regions=(N_CALO_REGIONS if region_np is not None else 0),
                     pos_feat_mean=(_m if pos_feat_np is not None else None),
                     pos_feat_std=(_s if pos_feat_np is not None else None),
                     **anchor_kwargs, **qt_kwargs, **ctx_kwargs, **set_kwargs).to(dev)
    if core_anchored != "none":
        anc = d["anchor"]; amode = d["anchor_mode"]
        print(f"core anchor: {core_anchored} (glob dims 2,3 are the residual); branches "
              + " ".join(f"{lab} {float((amode==c).mean()):.3f}" for c, lab in
                         [(0, "barrel"), (1, "endcap"), (2, "turning"), (3, "none")]) + "; "
              f"anchor phi std {anc[:,1].std():.3f}, residual core_phi std {d['glob'][:,3].std():.3f}")
    print(f"conditioning trunks: {'SEPARATE (3)' if args.separate_trunks else 'shared (1)'}")
    if width_norm:
        print(f"width_norm ON (slice): points are width-normalised, log_width = glob dim 4 "
              f"(mean {glob[:,4].mean():.2f} std {glob[:,4].std():.2f})")
    print(f"energy head: glob_idx={model.energy.glob_idx} (0=total_logE, 1=log_n)  "
          f"logE_max={logE_max:.3f} "
          f"(hardest cell {np.exp(logE_max):.4g} GeV)")
    print("  per-class logE_max: " + ", ".join(
        f"{int(c)}:{logE_max_pdg[int(c)]:.2f}" for c in np.unique(pt_pdg_np)))

    # standardise & move to GPU. Points: positions (d_eta,d_phi) standardised for
    # the flow; log-E kept in PHYSICAL units for the energy head + floor.
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg, dtype=torch.long, device=dev)
    with torch.no_grad():
        globS = model.std_glob(torch.as_tensor(glob, dtype=torch.float32, device=dev))
        posS = model.std_pos(torch.as_tensor(pts[:, :pos_dim], dtype=torch.float32, device=dev))
    logE = torch.as_tensor(pts[:, -1], dtype=torch.float32, device=dev)
    psh = torch.as_tensor(point_shower, device=dev)
    pt_contS, pt_pdg, pt_globS = contS[psh], pdgT[psh], globS[psh]
    # per-CELL cell count of the shower that cell belongs to, for the floor-vs-n head. Taken from the
    # slice offsets (RAW count), not recovered from the standardised log_n, so training and the
    # generation path (which passes the sampled `n[src]`) mean the same integer.
    pt_nT = torch.as_tensor(npt.astype(np.int64)[point_shower], device=dev)
    pos_featT = (torch.as_tensor((pos_feat_np - _m) / _s, dtype=torch.float32, device=dev)
                 if pos_feat_np is not None else None)
    regionT = (torch.as_tensor(region_np, dtype=torch.long, device=dev)
               if region_np is not None else None)
    with torch.no_grad():
        ancT = (model.anchor_feats(d["anchor"], d["anchor_mode"]).to(dev)
                if model.anchor_cond else None)
    tr_pt_idx = torch.as_tensor(np.where(tr_pt)[0], device=dev)
    tr_sh_idx = torch.as_tensor(np.where(tr_sh)[0], device=dev)
    va_pt_idx = torch.as_tensor(np.where(~tr_pt)[0], device=dev)
    va_sh_idx = torch.as_tensor(np.where(val_sh)[0], device=dev)

    # flow target: standardised positions, plus standardised log-E under --joint_energy
    flowX = posS if not args.joint_energy else torch.cat(
        [posS, ((logE - je_mean) / je_std)[:, None]], dim=1)

    # SHOWER-LEVEL BATCHES for the set model. The historical loop draws random POINTS from the whole
    # dataset, so two cells of one shower essentially never share a batch -- a set model would never
    # see a shower. Batches are drawn from size classes (powers of two) so padding waste stays < 2x,
    # and a class is chosen with probability proportional to its CELL count, which keeps each cell's
    # chance of being trained on the same as under the per-point sampler.
    off_t = torch.as_tensor(off[:-1], device=dev)
    npt_t = torch.as_tensor(npt, device=dev)

    def make_sampler(sh_ids):
        sz = npt[sh_ids]
        cls = np.floor(np.log2(np.maximum(sz, 1))).astype(int)
        groups = []
        for c_ in np.unique(cls):
            ids = sh_ids[cls == c_]
            groups.append((torch.as_tensor(ids, device=dev), int(npt[ids].max()), float(npt[ids].sum())))
        w = np.array([g[2] for g in groups]); w = w / w.sum()

        def draw():
            ids, nmax, _ = groups[int(rng.choice(len(groups), p=w))]
            B = max(1, args.set_budget // nmax)
            sh = ids[torch.randint(len(ids), (B,), device=dev)]
            n_b = npt_t[sh]
            rows = torch.repeat_interleave(torch.arange(B, device=dev), n_b)
            cols = torch.arange(rows.numel(), device=dev) - torch.repeat_interleave(torch.cumsum(n_b, 0) - n_b, n_b)
            return off_t[sh][rows] + cols, rows          # flat point indices, local shower index
        return draw

    draw_tr = make_sampler(np.where(tr_sh)[0]) if args.point_attn else None
    draw_va = make_sampler(np.where(val_sh)[0]) if args.point_attn else None

    def flow_loss(val=False):
        if args.point_attn:
            pidx, lsrc = (draw_va if val else draw_tr)()
            return model.points.cfm_loss(flowX[pidx], model.cond_embed(pt_contS[pidx], pt_pdg[pidx], "points"),
                                         pt_globS[pidx][:, :2], src=lsrc)
        idx = (va_pt_idx if val else tr_pt_idx)
        pi_ = idx[torch.randint(len(idx), (args.pt_batch,), device=dev)]
        return model.points.cfm_loss(flowX[pi_], model.cond_embed(pt_contS[pi_], pt_pdg[pi_], "points"),
                                     pt_globS[pi_][:, :2])

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
            cfm = flow_loss(val=True).item()
            eh = 0.0 if args.joint_energy else model.energy.loss(model.cond_embed(pt_contS[pi], pt_pdg[pi], "energy"),
                                   logE[pi], model.log_floor, glob_std=pt_globS[pi],
                                   n_cells=pt_nT[pi],
                                   pos_feat=None if pos_featT is None else pos_featT[pi],
                                   region=None if regionT is None else regionT[pi]).item()
            si = va_sh_idx[torch.randint(len(va_sh_idx), (args.glob_batch,), device=dev)]
            g = model.glob.nll(model.cond_embed(contS[si], pdgT[si], "glob"), globS[si],
                               None if ancT is None else ancT[si]).item()
        model.train()
        return cfm, eh, g

    t0 = time.time()
    for step in range(1, args.steps + 1):
        pi = tr_pt_idx[torch.randint(len(tr_pt_idx), (args.pt_batch,), device=dev)]
        si = tr_sh_idx[torch.randint(len(tr_sh_idx), (args.glob_batch,), device=dev)]
        cfm = flow_loss()
        ehl = torch.zeros((), device=dev) if args.joint_energy else model.energy.loss(model.cond_embed(pt_contS[pi], pt_pdg[pi], "energy"),
                                logE[pi], model.log_floor, glob_std=pt_globS[pi], n_cells=pt_nT[pi],
                                pos_feat=None if pos_featT is None else pos_featT[pi],
                                region=None if regionT is None else regionT[pi])
        gnll = model.glob.nll(model.cond_embed(contS[si], pdgT[si], "glob"), globS[si],
                              None if ancT is None else ancT[si])
        loss = cfm + ehl + args.glob_weight * gnll
        opt.zero_grad(); loss.backward(); opt.step()

        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            vc, ve, vg = val_loss()
            print(f"  step {step:6d} | cfm {cfm.item():.4f} | ehl {ehl.item():.4f} | gnll {gnll.item():.4f} "
                  f"| val_cfm {vc:.4f} val_ehl {ve:.4f} val_gnll {vg:.4f} | {sps:.1f} it/s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"train/cfm": cfm.item(), "train/ehl": ehl.item(), "train/gnll": gnll.item(),
                           "val/cfm": vc, "val/ehl": ve, "val/gnll": vg}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = Path(args.out) / f"checkpoint_{step:06d}.pt"
            torch.save({"model": model.state_dict(), "norm": norm, "args": vars(args), "step": step}, ckpt)
            print(f"  saved {ckpt}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
