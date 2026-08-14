# 2026-08-13 — e± capacity test (partial) + width-augmented gate (reframes everything)

- **Date:** 2026-08-13 · **commit:** 9e0bbc2 (working tree) · **branch:** flow-response
- Jobs: 12326020 (e± capacity), 12326021 (width-gate baselines). Follows
  [2x2 + multispecies](2026-08-13-calo-energy-fix-2x2-and-multispecies.md).

## Hypothesis
1. **Capacity:** e± are 48% of calo energy / 57% of cells; the shared 17-class head gives e- 0.888
   while the single-species photon head reaches 0.557 on the same architecture. If the gap is shared
   capacity, a dedicated e± model on the SAME data (400k/class) and SAME 60k steps should land near
   photon-like 0.56-0.62.
2. **Width blind spot:** the 8 gate features are all energy/multiplicity, so the pion's and proton's
   bad `shower_width` (W/σ 0.32/0.33) could not move the gate. Adding it should expose real defects.

## Change
- `calo_metrics.py`: `WIDTH_FEATURE_NAMES = [width_mean, width_std]` — per-shower widths of the
  showers in each event, aggregated to event level. Reported as a **separate 10-feature gate**
  (`event_gate_auc_width`) so the historical 8-feature series (`event_gate_auc`) stays comparable.
  Per-feature single-variable AUCs now cover the width columns too.
- New slice `electron_v1.npz` (pdg 0,1, 400k/class, 800k showers / 8.5M points) and checkpoint
  `electron_v1` (2 classes, 60k steps, otherwise identical recipe to `multispecies_v1`).

## Result 1 — capacity is a REAL but PARTIAL lever
| run | gate8 | gate10 | cell_logE | width W/σ |
|---|---|---|---|---|
| e- single-species (2 classes) | **0.773** | 0.971 | 0.017 | 0.171 |
| e- multi-species (17 classes) | 0.888 | 0.980 | 0.022 | 0.177 |
| e+ single-species | **0.760** | 0.968 | 0.017 | 0.169 |
| e+ multi-species | 0.887 | 0.977 | 0.018 | 0.177 |

A dedicated e± head recovers **~0.12** of gate (0.888 → 0.773) — so shared capacity/interference is
real and worth spending on — but it does NOT reach photon-like 0.56. **e± are intrinsically harder
than photons** (median 9 cells/shower vs 4, and they are largely tracker secondaries with a wide
production-radius spread). So: capacity buys ~0.12; the rest is architectural.

## Result 2 — THE WIDTH GATE IS THE HEADLINE. Lateral shape is the dominant defect everywhere.
| run | gate8 | gate10 | jump | width_mean AUC | width_std AUC | frac_near_floor AUC |
|---|---|---|---|---|---|---|
| photon (the "good" one) | 0.558 | **0.805** | +0.247 | 0.584 | **0.689** | 0.510 |
| pion | 0.809 | **0.993** | +0.184 | **0.945** | **0.963** | 0.666 |
| e- multi-species | 0.888 | 0.980 | +0.092 | 0.748 | **0.914** | 0.702 |
| e+ multi-species | 0.887 | 0.977 | +0.090 | 0.739 | 0.908 | 0.695 |
| e- single-species | 0.773 | 0.971 | +0.198 | 0.762 | 0.908 | 0.616 |
| e+ single-species | 0.760 | 0.968 | +0.208 | 0.753 | 0.903 | 0.616 |

`width_std` — the shower-to-shower VARIATION in lateral width within an event — is a near-perfect
discriminator (pion **0.963**, e± **0.90-0.91**, photon 0.689), beating `frac_near_floor` (the
previous prime suspect) in every single case. **The photon's 0.557 was substantially an artifact of
a blind metric: with width it is 0.805.** Every species is badly wrong on lateral shape.

Direction of the defect (CPU check, 15-20k showers, sampled vs slice truth):

| per-shower width | q10 | q50 | q90 | std |
|---|---|---|---|---|
| real e± | 0.0032 | 0.0228 | 0.1733 | 0.261 |
| gen e± | 0.0048 | 0.0293 | 0.3507 | 0.206 |

Real widths span a factor ~50 from q10 to q90; generated are shifted too wide in the bulk (median
1.3x, q90 2x) yet **compressed overall** (std 0.79x real; photon 0.80x). The cause is the i.i.d.
point assumption: cells are drawn independently given (cond, global), and cond cannot predict a
shower's compactness, so every shower of similar energy gets a similar width instead of sampling
from a 50x-wide distribution.

## Verdict
- **KEPT:** the width-augmented gate, as the metric that actually tracks shower quality. Both AUCs
  are reported; the 8-feature number is retained only for comparability with pre-2026-08-13 runs.
- **KEPT (partial):** capacity as a lever for e± (~0.12), worth a widening sweep, but not the fix.
- **REPRIORITISED:** lateral width now outranks floor fraction as the top defect. The floor finding
  stands (`frac_near_floor` 0.62-0.70) but is second.

## Next
1. **Per-shower width in the GlobalHead** — the direct analogue of the core fix that already worked.
   Add `log_width` as a global dim (quantile-normalised, so its inverse is bounded), and at
   generation SCALE the sampled point cloud by the sampled width (train on width-normalised deltas).
   The core removed per-shower LOCATION variance from the point flow; this removes per-shower SCALE
   variance the same way, and makes the width distribution match by construction rather than by
   hoping an i.i.d. flow reproduces a 50x spread.
2. Then the floor-fraction global dim (previous plan, still valid, now second priority).
3. Capacity sweep for the shared multi-species head (wider heads / per-class output layers).
4. Incidence head still missing (P(deposits | particle), AUC 0.897 probe) — blocks an honest
   full-event calo gate.
