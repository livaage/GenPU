# Tracker v3: the "surface-local" frame is NOT rotated, and the position heads are independent

**Date** 2026-09-07 · **Commit** `32d0cab` · **Branch** `flow-response`
**Measurement only — real data, no model, no job** (login node, `surface_pion_proto.npz`, 809,750 hits).

## Hypothesis

Raised in conversation, not by a metric: v3 predicts a module token and then four per-module
standardised position tokens `(x, y, z, time)`. If those four heads are conditionally independent
given the decoder state, the model cannot represent within-hit position correlation. Expectation
before measuring: some correlation exists and the independence costs a little.

## Change

None to the code. Two things were checked by reading and then measuring:

1. **What "local" means.** `ModuleGeometry.local_residual` (`src/genpu/module_geometry.py:52`) is
   `(physical - module_mean) / module_std` on **global Cartesian axes**. There is no rotation into
   a module frame. The docstring's "LOCAL Cartesian residuals" means *centred and scaled*, not
   *module-frame*.
2. **How the heads are sampled.** `TrackerModuleARModel.generate`
   (`src/genpu/models/tracker_module_ar.py:265-271`): `layer` then `surface` IS hierarchical
   (the surface head is masked by the sampled layer), but `x_head`, `y_head`, `z_head`,
   `time_head` all read the SAME hidden state `last` and are sampled independently — and are NOT
   conditioned on the module that was just drawn. The module enters only afterwards, through
   denormalisation `resid * mod_std[module] + mod_mean[module]`.

Then measured the real within-module structure, and simulated the independent draw by permuting
each standardised column within its module (which is exactly what four independent categoricals do
to the joint, while leaving every marginal untouched).

## Result

**A silicon module is a PLANE, so the three residuals are linearly dependent by construction.**
Within-module correlation of the standardised residuals, modules with >= 200 hits:

| volume | type | \|corr(x,y)\| | median module std (x, y, z) mm |
|---|---|---|---|
| 17 | barrel | **1.000** | 3.39, 3.48, 20.83 |
| 24 | barrel | — | 9.81, 9.80, 31.21 |
| 29 | barrel | — | 19.57, 19.56, 0.53 |
| 16, 18 | endcap | 0.58 | 14.4, 14.4, **0.05** |
| 23, 25 | endcap | — | 35.3, 36.1, **0.05** |
| 28, 30 | endcap | — | 24.7, 24.7, **0.05** |

- **Barrel `corr(x,y)` is EXACTLY 1.000.** A stave is a plane tangent to the cylinder, so global
  δx and δy are the same tangential coordinate scaled two ways.
- **Endcap local z is degenerate** — std **0.05 mm**, i.e. a disc at fixed z. Its correlations are
  undefined (NaN), and the 512-way `z_head` is predicting a constant for those hits.

**Off-plane error from independent sampling.** Per barrel module: fit the plane of the real hit
cloud (smallest singular vector), measure the perpendicular spread, then repeat after permuting
each standardised column within the module.

| volume | share of hits | real off-plane spread | independent-column draw |
|---|---|---|---|
| 17 | 25.9% | **0.0000 mm** (p99 0.0000, 1346 modules) | **2.35 mm** |
| 24 | 20.6% | **0.0000 mm** (p99 0.0000, 1005 modules) | **6.54 mm** |

Real hits lie EXACTLY on the silicon — position smearing is in-plane only, which is correct for a
strip/pixel sensor (`true_x/y/z` vs `x,y,z` differ in-plane). Independent sampling scatters them
**2.4-6.5 mm off the plane**, against an in-plane module extent of 3.4 mm (vol 17). Barrel is
**52.1%** of hits (volumes 17/24/29 = 0.259/0.206/0.057); volume 29 had no module with >= 50 hits
in the pion proto slice, so its number is unmeasured but the geometry argument is identical.

## Verdict

**Real defect, previously unrecorded.** Generated barrel hits do not lie on any detector surface.

It survived because every gate feature is a marginal in `(r, phi, z)` and layer spacing is tens of
mm, so a few-mm radial smear is invisible to the current metric. It will NOT survive ACTS — reco
requires hits on surfaces — and the reco-level eval is the open item that decides v3 vs v4.

Note what is NOT claimed: no model was sampled here. The permutation test shows what the
factorisation *permits*, exactly, by destroying the joint while preserving every marginal. A
trained model could in principle place nearly all mass on one bin per axis and stay near the plane;
that has not been measured, and doing so would require it to encode the plane constraint in the
decoder state at every step rather than being given it.

## Next

The fix is NOT within-step autoregression over `(x, y, z)`. It is to **rotate into a genuine module
frame and predict two IN-PLANE coordinates, dropping the normal direction entirely**:

- makes the plane constraint exact by construction rather than something to be learned;
- removes the 512-way head that currently predicts a constant for the **47.9%** of hits in the
  endcaps;
- same class of change as v1 -> v3 surface-local (matched pion gate 0.9996 -> 0.80), and cheaper —
  `build_module_geometry.py` already has the per-module hit clouds needed to fit each frame.

Requires a rebuild of `module_geometry.npz` (add per-module rotation), a rebuild of the surface
slices, and a retrain. Existing v3 checkpoints stay loadable only if the frame is versioned in the
npz.

Second, smaller item found on the way: the module embedding is not an input to the position heads
at all. Whether that matters is untested and is a separate question from the frame.
