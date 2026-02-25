# Wave 5B.2 Diagnosis (2026-02-23)

## Scope

This note summarizes why current Goose `w5b2_*` runs show:

- rising `train_acc`
- falling `train_loss`
- decreasing `val_acc` after ~epoch 35-40

and whether this indicates an implementation bug.

## Runs Analyzed

- `runs/rml2016_goose/w5b2_spec_a020_h020`
- `runs/rml2016_goose/w5b2_oraclegate_a020_h020`
- comparison baselines:
  - `runs/rml2016_goose/w5b_moe2_learned_feat_bal001`
  - `runs/rml2016_goose/w5a_bidir2_cls256`

## Key Findings

### 1) Pattern is real overfitting, not expected late-stage convergence

For `w5b2_spec_a020_h020`:

- best `val_acc = 0.6253` at epoch 37
- epoch 40 -> 65:
  - `val_acc: 0.6242 -> 0.5966` (down `0.0275`)
  - `train_loss: 1.4624 -> 0.8482` (down `0.6142`)
  - `train_acc: 0.7330 -> 0.9014` (up `0.1684`)

For `w5b2_oraclegate_a020_h020`:

- best `val_acc = 0.6112` at epoch 34
- epoch 40 -> 64:
  - `val_acc: 0.6088 -> 0.5812` (down `0.0276`)
  - `train_loss: 1.3317 -> 0.8050` (down `0.5267`)
  - `train_acc: 0.7683 -> 0.9316` (up `0.1633`)

This is classic train/val divergence and is stronger than prior waves.

### 2) W5B.2 underperforms prior W5/W4 baselines even at peak

- `w5b2_spec_a020_h020`: peak `val_acc = 0.6253`
- `w5b2_oraclegate_a020_h020`: peak `val_acc = 0.6112`
- prior blind baselines:
  - `w5b_moe2_learned_feat_bal001`: peak `0.6409`
  - `w5a_bidir2_cls256`: peak `0.6405`

Gap from best prior blind run:

- W5B.2 spec: `-0.0156`
- W5B.2 oracle-gate: `-0.0297`

### 3) Oracle-gated MoE diagnostic failed (important)

`w5b2_oraclegate_a020_h020` uses oracle routing but still underperforms.

Interpretation:

- routing noise is not the primary failure mode
- current MoE-head specialization objective does not improve representation quality
- shared trunk remains bottleneck for low-SNR separation

### 4) Primary mechanism: head-specific CE is too aggressive in this setup

W5B.2 adds expert-specific CE on top of blended CE.
Observed effect:

- model reaches moderate validation accuracy earlier
- but train accuracy rises too fast and keeps rising
- generalization collapses after early peak

At similar validation level (~0.620):

- baseline W5B/W5A reaches it around epoch 36-37 with tiny train-val gap (~+0.003)
- W5B.2 reaches it at epoch 26 with much larger train-val gap (~+0.040)

This indicates early memorization pressure rather than healthy specialization.

### 5) No evidence of coding breakage from metrics behavior

Metrics indicate expected MoE bookkeeping:

- expert loads are stable and non-collapsed for 2 experts
- loss terms are finite and trend smoothly
- no NaNs, no unstable oscillations, no abrupt metric corruption

So this looks like an optimization/objective design issue, not a runtime bug.

## Secondary Observations

- low-band metrics did not improve enough to justify the high/mid degradation.
- even oracle-conditioned model ceiling remains below 0.70:
  - `w5a_oracle_bidir3_cls512` peak ~`0.6866`

This confirms the overall challenge is representation ceiling + blind gap, not just gate behavior.

## Conclusion

Current W5B.2 implementation is functionally correct, but the training objective is over-regularizing toward expert memorization and reducing generalization. The MoE-head path (as configured) is not producing a net gain.

## Recommended Next Actions

1. Stop/early-stop current W5B.2 runs after sustained post-peak decline.
2. Revert to best blind anchor (`w5b_moe2_learned_feat_bal001` / `w5a_bidir2_cls256`) for comparisons.
3. Pivot effort from MoE-head specialization to blind-oracle gap reduction on shared trunk:
   - stronger representation learning (pretraining or trunk upgrade), or
   - carefully controlled KD/curriculum variants if staying in current trunk family.
4. Keep MoE only as a lightweight, non-degrading variant unless a new objective proves a clear low-band win without high-band tax.

---

## Addendum (2026-02-25): Post-W5B.3/W5C2 Decision

Follow-up results after this diagnosis:

- W5B.3 deconfounding confirmed the train/eval oracle routing mismatch was real and fixed.
- Head-specific CE variants (detach and non-detach) still failed to produce a deployable gain over no-aux MoE baseline.
- W5C-SSLv2 pilots (MoCo + SupCon redesign) underperformed anchor and did not satisfy promotion gates.

Interpretation now:

1. The original W5B.2 failure call (overfitting/objective mismatch) was correct.
2. The MoE specialization lever is currently exhausted for this trunk family under blind deployment constraints.
3. The next best lever is temporal representation architecture, not additional SSL/loss tuning.

Current active direction:

- Wave 5D no-MoE trunk pivot with interchangeable CLDNN temporal backbones:
  - `lstm` (legacy control)
  - `tcn`
  - `resnet1d`
- Keep training recipe fixed and isolate trunk effects via an 8-run pilot (2 Goose + 6 Athena).

---

## Addendum (2026-02-26): Wave 5E Diffusion Front-End Implementation

Based on post-W5D.3 discussions, a deconfounded Wave 5E implementation is now in code:

1. Waveform diffusion front-end is added for CLDNN (`--dn-diff-enable`) with:
   - `v/eps` target support,
   - DDIM and one-step eval modes,
   - split train/eval `t_start` source,
   - SNR→t scale/bias calibration.
2. E.1 safety controls are implemented:
   - matched frozen-control capability (`--dn-diff-freeze-classifier`),
   - hard high-SNR bypass,
   - low-band-only train loss masking.
3. Diagnostics required by the revised plan are implemented:
   - conditioning ablation (`none|zero|shuffle`),
   - one-step vs DDIM comparison support,
   - `metrics.jsonl` fields for diffusion losses, t-start stats, and per-band active fractions.

Decision policy remains unchanged: run Wave 5E.1 first, then enter Wave 5E.2 only if E.1 passes hard gates.
