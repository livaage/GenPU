# First calo gate on cells that exist: projection is mechanically sound, the gate is 0.010 WORSE, and a pre-existing `wrap_pi` bug was corrupting every width number

**Date** 2026-09-15 · **Commit** `32d0cab` (+ `src/genpu/calo_cells.py`, `--snap_cells`) · **Branch** `flow-response`
**Jobs** 13928711 (contaminated, see below), **13929818** (the result, 13m51s)
ckpt `multispecies_v2_s0/checkpoint_060000` · all 17 classes · `--max_cells 4096` both arms · 1 seed (paired A/B)

## A PRE-EXISTING BUG, found by this experiment but not caused by it

`calo_metrics.py` computed `g_dp = gen_phi - p_phi[gen_src]` with **no wrap**, while the real side
has always been wrapped (`build_calo_slice_v2.py:292`, `wrap_pi(c_phi - (p_phi + AP[k]))`).

I first assumed wrapping would be a no-op on the unsnapped path, since `gen_phi` is the unwrapped sum
`p_phi + core + pos` and cell offsets are tiny. **That was wrong, and the fix moved the PLAIN arm
too**: the shower CORE is not tiny -- the helix anchor displaces pion cores by ~1.5 rad -- so
`core + pos` can exceed pi and the unwrapped difference then disagreed with a wrapped reference.

| plain arm | before wrap fix | after |
|---|---|---|
| `width_mean` AUC | 0.5618 | **0.5100** |
| `width_std` AUC | 0.5016 | **0.6953** |
| `d_phi` W/sigma | 0.0343 | 0.0358 |
| `shower_width` W/sigma | 0.0344 | 0.0505 |
| `event_gate_auc_width` | 0.9271 | 0.9299 |

**Every previously reported width-gate and d_phi number is affected**, in both directions --
`width_mean` was pessimistic, `width_std` flatteringly optimistic (0.50 is "indistinguishable"; the
truth is 0.70). The width gate itself barely moves, so headline conclusions survive, but any
argument that leaned on `width_std` being at chance should be re-read.

How it was caught: with `--snap_cells` the bug became enormous (`width_std` 0.5016 -> **0.9565**,
`d_phi` W 0.0343 -> **0.4025**) because snapping recomputes phi as `arctan2(y, x)` in (-pi, pi].
**It looked exactly like a physics regression** -- "cell projection destroys shower width" -- and was
one missing `wrap_pi`. The tell was that `d_eta` was untouched (0.0318 -> 0.0320) while `d_phi` blew
up 12x: a break on one of two symmetric coordinates is a coordinate bug, not physics.

## The projection itself is mechanically sound

    42,109,049 points -> 40,587,800 cells (merged 3.61%)
    energy conserved x1.000001
    cells with >1 contributor 0.0309

## Result — the gate is slightly WORSE

Compare `event_gate_auc` (8 features, identical on both arms). **The decomposition's "full" numbers
are NOT comparable between arms** -- plain carries 12 features, snap 10, because depth observables
are dropped under `--snap_cells` (the snapped depth is a layer index, not the continuous coordinate
the real side carries). Reading 0.9638 -> 0.9486 as an improvement would be a feature-count artifact.

| | plain | snap |
|---|---|---|
| **`event_gate_auc` (comparable)** | **0.9357** | **0.9458** |
| `event_gate_auc_width` | 0.9299 | 0.9398 |
| marginals only | 0.7879 | 0.8652 |
| copula only | 0.7772 | 0.7873 |

| per-feature AUC | plain | snap | delta |
|---|---|---|---|
| `logE_p90` | 0.6098 | 0.7158 | **+0.106** |
| `logE_mean` | 0.5238 | 0.6017 | **+0.078** |
| `logE_std` | 0.7074 | 0.7602 | **+0.053** |
| `frac_near_floor` | 0.5081 | 0.5351 | +0.027 |
| `cells_per_src` | 0.5123 | 0.5338 | +0.022 |
| `n_cells` | 0.5047 | 0.5044 | -0.000 |
| `width_mean` / `width_std` | 0.5100 / 0.6953 | 0.5118 / 0.6953 | ~0 |

W/sigma `cells_per_shower` **0.0104 -> 0.0202** (2x worse); positions unchanged
(`d_eta` 0.0318 -> 0.0320, `shower_width` 0.0505 -> 0.0512).

**The cost is entirely ENERGY.** Merging sums co-occupant energies, so per-cell energy rises; the
width and position marginals are untouched, which is what a correct projection should do.

## The pre-registered prediction was HALF right, and the half it got wrong is informative

Recorded before the run: merging removes cells so `n_cells`/`cells_per_src` fall and energies rise
(**correct**); the gate improves if co-occupancy is realistic and worsens if too many points pile
into one cell, with `multi_contrib_frac` vs ~3.5% deciding (**wrong dichotomy**).

Co-occupancy came out 3.09%, and the gate got worse anyway. The dichotomy missed the possibility
that the generator **UNDER-merges**: the right comparison is not the 3.5% cross-particle
superposition but the **7.2% WITHIN-shower sharing** (2026-08-24 cell-sharing entry), because the
merge is keyed on `src`. At 3.09% against 7.2% the generated points are **too spread out to land in
the same cell**, so real showers concentrate more energy per cell than generated ones -- which is
exactly the direction the energy marginals moved.

## Verdict — KEEP the projection, and read the regression as a NEW measurement

The projection is correct (energy exact, positions untouched, real cells snap to themselves at
0.99698 within 10 um and recover the same cell id 0.9925 of the time). The +0.010 gate cost is not a
flaw in it; it is the projection **exposing a defect that the continuous representation hid**: a
continuous cloud can put two points 0.1 mm apart and be scored as two cells, where the detector
would report one. Cells are the measurement the detector actually makes, so this is the more honest
number, and 0.9357 was flattering.

It also connects to the standing coherence result: "cells too spread out to share a channel" and
"generated showers have no two-point coherence, gen xi flat vs real 1.54-2.22 for hadrons" are
plausibly the same defect seen at two scales. NOT verified.

## Next

- **Co-occupancy vs distance from the shower core.** If the generator under-merges because its core
  is too diffuse, the deficit should be concentrated in the core. One cheap probe, no retrain.
- Re-run the width-sensitive conclusions that predate the wrap fix.
- Boundary assignment (HCAL endcap 2.5%, barrel 0.2-0.5%) is still unfixed and is a separate, smaller
  source of wrong cells.
- A cell-level metric proper: occupancy and energy-per-cell against real cells, now possible.
