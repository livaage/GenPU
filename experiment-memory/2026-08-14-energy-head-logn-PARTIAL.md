# 2026-08-14 — energy head on `log_n`: hypothesis PARTLY right, insufficient; anchor thread closed

- **Commit / branch**: `433edcc` / `flow-response`
- **Jobs**: 12380210 (arm A, shared trunk + log_n), 12380211 (arm B, separate trunks + anchor_cond
  + log_n). Both COMPLETED ~28 min.
- Follows [the falsification entry](2026-08-14-anchor-cond-separate-trunks-FALSIFIED.md)

## Hypothesis

`frac_near_floor` is a function of multiplicity, which is the GlobalHead's variable; with
`--no_energy_glob` the energy head only ever gets it implicitly through a shared trunk. Give it
`log_n` explicitly (glob dim 1, not dim 0) and (A) the floor fraction improves, (B) trunk isolation
becomes free, so separate trunks + anchor_cond finally ships.

## Result

| e− config | gate8 | gateW | `frac_near_floor` | `cell_logE` | `logEreco` | val_ehl |
|---|---|---|---|---|---|---|
| **Phase 1 helix** | **0.7456** | **0.9639** | 0.6008 | 0.0153 | 0.0399 | 0.2083 |
| A: + log_n | 0.7781 | 0.9655 | **0.5833** | 0.0172 | **0.0165** | **0.2045** |
| acond + sep | 0.9648 | 0.9877 | 0.8282 | 0.0796 | 0.0317 | 0.2655 |
| B: acond + sep + log_n | 0.9662 | 0.9876 | 0.8574 | 0.0802 | 0.0319 | 0.2633 |

`frac_near_floor` with log_n, all three species: pion 0.6176 → **0.5744**, e− 0.6008 → **0.5833**,
e+ 0.5924 → **0.5637**.

## Reading — right mechanism, wrong magnitude

**(A) The hypothesis is directionally CONFIRMED and small.** log_n improves the floor fraction on
every species, improves per-shower reconstructed energy 2.4x (`logEreco` W/σ 0.0399 → 0.0165), and
improves the energy head's likelihood in both trunk settings (shared 0.2083 → 0.2045, separate
0.2655 → 0.2633). But it costs the upper energy percentiles (`logE_p90` 0.548 → 0.589, `logE_mean`
0.531 → 0.557), so the composite gate8 gets slightly WORSE (0.7456 → 0.7781). Net: not a win.

**(B) The hypothesis does NOT explain the separate-trunks damage.** log_n recovers essentially none
of it — val_ehl 0.2655 → 0.2633 against a shared-trunk 0.2083, `cell_logE` unmoved at 0.080. So the
energy head's loss under isolation is a **multi-task representation benefit**, not a missing
multiplicity signal: it does better sharing a trunk with the point and global tasks than alone,
independent of what it is told about the count.

A clean confirmation that separate trunks really do isolate: `acondsep` and `full` differ ONLY in the
energy head's inputs, and their val_gnll is bit-identical (-3.3952) as are `width_std` 0.9003,
`width_mean` 0.7540, `d_eta` 0.0078, `d_phi` 0.1451. Gradient isolation is exact.

The trade is also real and visible: separate trunks + anchor_cond give a much better GlobalHead
(val_gnll -3.395 vs -2.427) and a worse energy head (val_ehl 0.2655 vs 0.2083). The calo gate is
currently dominated by energy features, so that trade loses.

## Verdict — the anchor-conditioning thread is CLOSED

**Plain Phase 1 helix (`pion_anchor`, `electron_anchor`) remains the best config, unbeaten across
six variants tried today**: anchor_cond, `line`/`auto` anchors, anchor_cond+separate trunks,
energy-on-log_n, and all three combined. `--energy_glob_idx` stays in the code as an opt-in flag; it
is the best per-shower-energy setting on record (`logEreco` 0.0165) if that ever becomes the target.

The recurring pattern across all six: **every change improved its own mechanism target and lost on
the gate**, because the gate is dominated by energy/multiplicity features that sit downstream of a
head we keep perturbing. The core is now solved (e± per-bin spread 0.585 → 0.99 in the residual
frame, physical 1.275 → 1.00 with conditioning); the binding constraint has moved to the energy head.

## Next — Phase 2, as planned, and re-read the metric

1. **GlobalHead mixture → ShowerFlow.** Sharpened target: the PION residual is under-dispersed 0.93
   in a frame where e± sits at 0.99, so it is the pion's heavy-tailed conditional, not location.
2. **The energy head is now the binding constraint on the gate**, and no per-head patch has moved it.
   `frac_near_floor` has been the top or near-top discriminator in every run since 2026-08-13. It
   deserves its own structural treatment (the plan's floor-fraction-as-a-global idea), not another
   conditioning tweak.
3. Worth questioning the metric itself: 8 of 10 gate features are energy/multiplicity, so a change
   that halves the core error and nudges `logE_p90` reads as a regression. The width gate was added
   for exactly this reason; it may need the same treatment for the core.
