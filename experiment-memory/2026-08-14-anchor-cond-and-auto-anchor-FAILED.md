# 2026-08-14 — Anchor conditioning + `auto` (pT-split) anchor: both FAIL the gate

- **Commit / branch**: `9e0bbc2` / `flow-response` (working tree)
- **Jobs**: 12376190 (auto slice), 12376191 (anchor_cond), 12376192 (auto trainings)
- **Checkpoints**: `pion_anchor_cond`, `electron_anchor_cond`, `electron_auto`, `electron_auto_cond`
- Follows [the Phase 1 result](2026-08-14-phase1-helix-anchored-core-RESULT.md)

## Hypotheses

1. **Anchor conditioning**: Phase 1 measured that the real residual core scale differs 10x (pion) /
   15x (e±) between anchor branches, and the branch is a hard threshold the trunk's smooth features
   cannot express — so feed the GlobalHead `[a_eta, a_phi, |a|]` + a 4-way branch one-hot.
2. **`auto` anchor for e±**: helix below pT 0.035 GeV, straight line above. Pre-training probe
   (120k showers) gave pooled φ tightening **2.69x** vs 1.86x helix / 2.10x line, taking the better
   branch in every pT bin, with median |core| 0.302 → **0.069**. The bremsstrahlung fix.

## Result — every new config is WORSE than plain Phase 1 helix

e− (e+ tracks it within 0.01 throughout):

| config | gate8 | gateW | `width_std` | `cell_logE` W/σ | mechanism (physical) | (residual) |
|---|---|---|---|---|---|---|
| baseline `electron_v1` | 0.7729 | 0.9713 | 0.9085 | 0.0173 | 0.585 | — |
| **Phase 1 helix** | **0.7456** | **0.9639** | **0.8952** | 0.0153 | 1.275 | 0.992 |
| helix + anchor_cond | 0.8857 | 0.9719 | 0.8999 | 0.0186 | **1.003** | 0.996 |
| auto | 0.9679 | 0.9884 | 0.9044 | **0.0820** | 0.811 | 0.646 |
| auto + anchor_cond | 0.8798 | 0.9774 | 0.9114 | 0.0162 | 0.829 | 0.828 |

Pion (anchor_cond): gate8 0.8055 → 0.8155, gateW 0.9833 → 0.9828, `width_std` 0.9425 → 0.9416,
residual spread 0.931 → 0.952, branch ratios 1.13/0.93/0.92 → 1.08/0.87/0.94. Flat.

## Diagnosis 1 — anchor conditioning works, and pays for it through the SHARED TRUNK

It did exactly what it was designed to do: physical-frame mechanism **1.275 → 1.003**, branch ratios
1.17/1.48/0.85 → 0.87/0.74/0.93, and `d_eta` W/σ **0.0269 → 0.0067**. The cost lands on the *energy*
head — `frac_near_floor` AUC 0.601 → 0.660, `logE_mean` +0.043, `cell_logE` 0.0153 → 0.0186 — which is
the identical fingerprint to width_norm and ctx_norm on 2026-08-13.

The features were injected **after** the trunk specifically to avoid this, which sharpens the lesson:
that blocks FORWARD contamination but not BACKWARD. Handing the GlobalHead the anchor changes how
much it needs the trunk, so the gradient it returns changes what the energy head sees. **With a
shared trunk, any change to one head's task perturbs the others, even one injected downstream.**

## Diagnosis 2 — `auto` fails on a DISCONTINUITY the probe could not see

The frame really is 2.69x tighter, but the pT split makes the anchor rule jump at 0.035 GeV: two
showers with near-identical conditioning are anchored by different rules, and the GlobalHead
conditions on `log_pt` smoothly, so it blends two unlike residual distributions — residual spread
**0.646**, the worst of any run. A self-inflicted copy of the branch-blindness the other lever was
meant to fix, confirmed by `auto+cond` recovering to 0.828: the one-hot encodes barrel/endcap/turning
but **not** helix-vs-line, so it only partially informs the model. `cell_logE` **0.0820** (5x) is the
same shared-trunk channel, much louder because the GlobalHead's task got genuinely harder.

**Measuring the frame in isolation is not sufficient** — a tighter target that is a discontinuous
function of the conditioning can be harder to model than a looser smooth one. Every prior frame win
here (v3 surface-local, v4 helix, Phase 1) was smooth in the conditioning.

## Verdict

**Both ABANDONED as run.** Phase 1 plain helix (`pion_anchor`, `electron_anchor`) stays the best
calo config. `--anchor_cond` and `--core_anchor auto` remain in the code as opt-in flags.

`auto` is dropped rather than patched: feeding the anchor KIND as a feature would probably fix the
discontinuity, but that is a second epicycle on a lever whose isolated 2.69x frame gain has not
translated once.

## Next

**Retest anchor conditioning with `--separate_trunks`** — the diagnosed cause, and the config
Phase 0c already recommended for pion / e± (deferred in Phase 1 to keep the comparison
like-for-like). If interference is the cause, the 1.003 mechanism should survive while the gate
recovers. Risk to state up front: Phase 0c measured separate trunks helping e± on their own
(0.773 → 0.739), but the two effects need not compose.
