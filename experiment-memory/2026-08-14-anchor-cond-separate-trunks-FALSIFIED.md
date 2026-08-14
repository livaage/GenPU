# 2026-08-14 — anchor_cond + separate trunks: prediction FALSIFIED, and it unifies five failures

- **Commit / branch**: `9e0bbc2` / `flow-response` (working tree)
- **Job**: 12378125 (COMPLETED 00:27:48) · ckpts `pion_anchor_cond_sep`, `electron_anchor_cond_sep`
- Follows [anchor_cond FAILED](2026-08-14-anchor-cond-and-auto-anchor-FAILED.md)

## Hypothesis (stated in advance, in the job script)

Anchor conditioning fixed its mechanism target but degraded the untouched energy head; that was
diagnosed as shared-trunk interference (the gradient the GlobalHead returns changes what the energy
head sees). **Prediction: with `--separate_trunks` the 1.003 mechanism survives and the gate
recovers to at least the Phase 1 helix numbers.**

## Result — the mechanism survived; the gate got WORSE, not better

| e− config | gate8 | gateW | `cell_logE` W/σ | `frac_near_floor` | mechanism (phys) |
|---|---|---|---|---|---|
| **Phase 1 helix** | **0.7456** | **0.9639** | 0.0153 | 0.6008 | 1.275 |
| helix + cond (shared) | 0.8857 | 0.9719 | 0.0186 | 0.6600 | **1.003** |
| sep trunks only (Phase 0c) | 0.7385 | 0.9707 | 0.0160 | 0.5821 | — |
| **helix + cond + sep** | **0.9648** | 0.9877 | **0.0796** | **0.8282** | 1.014 |

Pion: gate8 0.8055 → 0.8176, gateW 0.9833 → 0.9840, mechanism 0.929 → **1.023**, `frac_near_floor`
0.6176 → **0.5868** (improved), `cell_logE` 0.0223 → 0.0258. Roughly flat on the gate.

**The trunk-interference diagnosis is wrong.** Isolating the trunks should have removed the damage;
it multiplied it (`cell_logE` 0.0186 → 0.0796, `frac_near_floor` 0.660 → 0.828).

## What is actually going on — one mechanism behind five failures

The energy head's validation loss is *worse* with separate trunks: **val_ehl 0.2655 vs 0.2141** at
60k, identical everywhere else (val_cfm 1.4053 vs 1.4055). So the energy head genuinely loses
information when isolated — it is not a sampling artifact.

`frac_near_floor` is a function of MULTIPLICITY, which is the GlobalHead's variable (`log_n`). These
runs use `--no_energy_glob`, so the energy head is never given multiplicity explicitly and must infer
it **implicitly through the shared trunk**. That single fact explains every failure in this family:

| change | what it does to the energy head's implicit multiplicity signal | `frac_near_floor` |
|---|---|---|
| width_norm (2026-08-13) | changes the GlobalHead task | 0.616 → 0.858 |
| ctx_norm (2026-08-13) | changes the GlobalHead task | 0.616 → 0.858 |
| anchor_cond | gives the GlobalHead a direct route, so it leans on the trunk less | 0.601 → 0.660 |
| `auto` anchor | same, plus a discontinuous target | (cell_logE 0.082) |
| **cond + separate trunks** | **removes the shared route entirely** | **0.601 → 0.828** |

It also predicts Phase 0c's species ordering, which was previously only noted as a correlation:
separate trunks hurt the photon (4.4 cells/shower), were mild on e± (10.6) and helped the pion
(15.6) — the fewer cells a shower has, the more the floor fraction matters per event and the more
the energy head depends on knowing multiplicity.

## Verdict

**Abandoned as run.** Plain Phase 1 helix stays the best gate config. But the anchor conditioning
result is now confirmed robust and trunk-independent: it fixes the core mechanism on both species
(e± physical 1.275 → 1.00, pion 0.929 → 1.02, branch ratios → ~0.8-1.0) at identical val_cfm/val_gnll.
The ONLY thing standing between it and a shipped win is the energy head's floor fraction.

## Next — the experiment Phase 0c wrote down and never ran

**Separate trunks + condition the energy head on `log_n` ONLY** (glob dim 1, not dim 0). Supplies
multiplicity explicitly instead of hoping it leaks through a trunk. This is NOT the ruled-out
"energy head on the sampled global": that failure was `total_logE` (dim 0) making continuous cell
energy a sharp function of a noisy energy scale. A count is a different quantity, and the floor
Bernoulli — not the continuous mixture — is what needs it.

Needs a small change: `EnergyHead.glob_dim` currently takes a PREFIX of the globals
(`glob_std[:, :glob_dim]`), so it cannot express "dim 1 only" — make it take an index list.
