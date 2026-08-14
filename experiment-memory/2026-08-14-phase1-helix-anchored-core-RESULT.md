# 2026-08-14 — Phase 1 RESULT: truth-helix anchored core (pion + e±, shared trunk)

- **Commit / branch**: `9e0bbc2` / `flow-response` (working tree)
- **Jobs**: 12374054 / 12374055 (slices + corrected Phase 0b), 12374056 (train + metrics + diag,
  COMPLETED 00:26:21)
- **Checkpoints**: `pion_anchor/checkpoint_040000`, `electron_anchor/checkpoint_060000`
- **Slices**: `pion_anchor.npz` (2.52M showers), `electron_anchor.npz` (800k)
- Builds on [the sign-bug entry](2026-08-14-helix-endcap-sign-bug-and-phase1-anchor.md)

## Hypothesis

Store the shower core as a residual from the truth-helix extrapolation to the calo front face; the
GlobalHead mixture then predicts a small local residual instead of a 1.5 m bending displacement.
Target: per-(charge × pT)-bin core spread ratio → ~1.0, from 0.58x (e±) / 0.94x (pion).

## Change

`--core_anchor helix` slices; shared trunk, and otherwise byte-identical recipes and step counts to
the references (`pion_qtd_v2` 40k, `electron_v1` 60k), so the core frame is the only variable.

## Result — every gate metric improves, on all three species

| | e− ref | **e− anch** | e+ ref | **e+ anch** | pion ref | **pion anch** |
|---|---|---|---|---|---|---|
| gate (8-feat) | 0.7729 | **0.7456** | 0.7597 | **0.7488** | 0.8092 | **0.8055** |
| gate (width) | 0.9713 | **0.9639** | 0.9675 | **0.9578** | 0.9931 | **0.9833** |
| `width_std` AUC | 0.9085 | **0.8952** | 0.9029 | **0.8905** | 0.9627 | **0.9425** |
| `width_mean` AUC | 0.7617 | **0.7330** | 0.7528 | **0.7270** | 0.9449 | **0.9081** |
| `shower_width` W/σ | 0.180 | **0.167** | — | 0.167 | 0.276 | **0.261** |

**No collateral damage** — `cell_logE` W/σ 0.0153 (e−) / 0.0223 (pion), `frac_near_floor` AUC 0.601 /
0.618, i.e. the energy head is untouched. This is the distinguishing feature vs the earlier patches:
`ctx_norm` bought a better pion `width_std` (0.837) but detonated the e± 8-feature gate
(0.773 → **0.967**), and `width_norm` did the same. The anchor is the first core change that helps
every species at once, which is what a frame change should do.

## Mechanism — read it in BOTH frames (the diag now reports both)

**e±: the mixture's conditional under-dispersion is FIXED.** In the residual frame — the quantity the
mixture actually fits — gen/real per-bin spread is **0.99** (min 0.97, max 1.01, all 12 bins), from
**0.585** pre-anchor. That was the defect four previous patches failed to move.

**Pion: not fixed. 0.93** in the residual frame (0.929 physical), vs 0.935 baseline / 0.931 ctx_norm.
Making the target 3.6x tighter left the *relative* under-dispersion untouched — it is a scale-free
property of the 8-component mixture, i.e. a density-model problem (**Phase 2, ShowerFlow**), not a
frame problem.

**New defect, both species — the GlobalHead is BLIND TO THE ANCHOR BRANCH.** Real residual scale by
branch, and what the model produces:

| branch | pion s_real → s_gen (ratio) | e± s_real → s_gen (ratio) |
|---|---|---|
| barrel | 0.057 → 0.064 (**1.13**) | 0.0086 → 0.0100 (**1.17**) |
| endcap | 0.057 → 0.053 (0.93) | 0.0104 → 0.0154 (**1.48**) |
| turning point | 0.598 → 0.551 (0.92) | 0.132 → 0.113 (0.85) |

The real scale differs **10x (pion) / 15x (e±)** between a face-reaching shower and a curler, and the
conditioning (`log_pt, eta, log_E, charge, mass, vr, vz` + PDG) cannot express the branch — it is a
hard threshold (2R vs r_calo; arc length vs πR). So the mixture splits the difference: face showers
come out too wide, curlers too narrow.

**e± physical frame now OVER-disperses (1.29), and it is the ANCHOR's fault, not the model's.** The
overshoot is monotone in pT — 0.96, 1.04, 1.13, 1.17, 1.45, **1.93** — matching the corrected Phase 0b
finding that the e± anchor gain degrades with pT and inverts (4.52x → **0.43x**). With the mixture at
0.99 in the residual frame, the residual physical error is entirely the anchor adding a bend that the
real deposit does not have. That is **bremsstrahlung**: photons radiated early travel straight from
the radiation point, so a stiff electron's own curved path is the wrong predictor and gets wronger the
further it bends. Branch (ii) of the plan's e± caveat.

## Verdict

**KEPT.** First core change that improves pion and e± simultaneously with no energy-head collateral;
fixes the e± mechanism outright (0.585 → 0.99).

## Next (evidence-ordered)

1. **Feed the anchor to the GlobalHead** — |anchor| + branch one-hot as extra conditioning dims.
   Deterministic truth, same status as the anchor itself, no exposure-bias risk. Directly targets the
   10-15x scale blindness measured above. Cheapest remaining lever.
2. **Straight-line anchor for e±** (q = 0 path, the code branch neutrals already take): predicted to
   beat the helix for stiff e± if brem is the mechanism, and it is a one-flag experiment. A pT-split
   anchor (helix soft / straight stiff) is the natural follow-up.
3. **Phase 2 ShowerFlow** now has a sharp, isolated target: the pion residual is under-dispersed 0.93
   in a frame where e± is at 0.99, so it is the pion's heavy-tailed conditional the mixture cannot
   fit, not the location.
4. The turning-point branch (63% pion / 73% e±) is energy booked to a parent that never reached the
   calo. It caps what any truth-only anchor can do and is exactly **Phase 4**'s population.
