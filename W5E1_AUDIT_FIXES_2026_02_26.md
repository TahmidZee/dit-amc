# Wave 5E.1 Audit + Fix Log (Post-Verification)

Date: 2026-02-26  
Scope: updates since `W5E1_VERIFICATION_2026_02_26.md`

---

## 1) Executive Summary

The diffusion implementation was mathematically correct at the DDPM/DDIM level, but several system-level training mismatches were suppressing downstream classification gains.

Since the last verification doc, the code has been updated to:

1. Remove/guard major train-vs-eval mismatches.
2. Add task-aware denoiser supervision (`lambda_dn_cls`) so denoiser outputs are optimized for classification utility.
3. Add strict guardrails for unsafe configurations (objective mismatch, unsupervised predicted-SNR control).
4. Reduce decoupled duplicate denoiser passes by reusing cached denoiser tensors from the classifier forward when possible.

The core residual limitation remains: diffusion target still reconstructs the noisy sample unless a cleaner supervisory target is introduced.

---

## 2) Findings Identified After Last Verification

## F1. Reconstruction target identity ceiling

Observed behavior:

1. `x_dn_ref` is taken from current noisy sample path in training.
2. Diffusion objective predicts reconstruction relative to this noisy reference.
3. Best achievable solution under pure diffusion loss is near-identity to noisy input.

Impact:

1. Denoiser can converge while showing little or no classifier improvement.
2. Flat validation despite decreasing diffusion loss is expected in this regime.

## F2. Conditioning distribution mismatch (train loss vs eval)

Observed behavior:

1. Training diffusion loss could use extra-degraded condition while eval used raw condition.
2. This introduced avoidable conditioning shift.

Impact:

1. Unstable transfer from training objective to eval denoiser behavior.

## F3. Decoupled duplicate denoiser passes

Observed behavior:

1. One pass inside model forward for classifier path.
2. Another independent pass for diffusion loss block.

Impact:

1. Additional compute.
2. Potential objective drift when passes use different sampled trajectories.

## F4. No mandatory task-aware signal in frozen-classifier mode

Observed behavior:

1. Frozen-classifier runs could rely only on reconstruction-style objectives.

Impact:

1. Denoiser optimized for waveform reconstruction, not decision-boundary utility.

## F5. Predicted-SNR control could run with no active supervision

Observed behavior:

1. Predicted SNR controls (t-start/bypass) could be active with insufficient explicit noise-head supervision in some settings.

Impact:

1. Control path quality could drift or become untrustworthy in long runs.

---

## 3) Fixes Implemented in Code

## A. Objective/path guardrails

1. Default eval mode changed to `onestep` for parity with one-step train forward.
2. Hard guard added: one-step-train + DDIM-eval now errors unless explicitly overridden.
3. Same mismatch guard added in eval-only mode.

Primary references:

1. `train.py` CLI: `--dn-diff-eval-mode` default `onestep`
2. `train.py` guard: objective mismatch block
3. `run_eval` guard: same block for eval-only

## B. Task-aware denoiser supervision

1. Added `--lambda-dn-cls` loss.
2. Applies CE on denoised outputs (`x0_dn`) and adds to total loss.
3. Works with mixup/non-mixup paths and curriculum mask.

Primary references:

1. `train.py` CLI: `--lambda-dn-cls`
2. `train.py` dn-diff loss block: CE on denoised output and accumulation.
3. `metrics.jsonl` now logs `lambda_dn_cls`, `train_loss_dn_cls`.

## C. Conditioning shift control

1. Added `--dn-diff-loss-cond-source {raw,degraded}`.
2. Default set to `raw` to match eval-time condition distribution.
3. `degraded` remains available for explicit ablation.

Primary references:

1. `train.py` CLI: `--dn-diff-loss-cond-source`
2. dn-loss block: conditional source switch.

## D. Duplicate denoiser-pass mitigation

1. Model forward now caches one-step denoiser tensors:
   1. prediction
   2. target
   3. reconstructed x0
   4. t-start
2. Training loss block reuses cached tensors when shape-compatible.
3. Falls back to recompute only when cache is unavailable/incompatible.

Primary references:

1. `model.py`: `_dn_diff_train_pred_flat`, `_dn_diff_train_target_flat`, `_dn_diff_train_x0_flat`.
2. `train.py`: cache capture helper and reuse path.

## E. Predicted-SNR supervision guardrails

1. Added required-supervision guard (default enabled) for predicted-SNR control path:
   1. If predicted controls are active and `lambda_noise <= 0`, run errors by default.
2. Override exists for debug only.

Primary references:

1. `train.py` CLI: `--dn-diff-require-noise-supervision` / disable override.
2. `train.py` validation guard block.

## F. Eta detach made explicit and configurable

1. Added `--dn-diff-detach-eta-cond` (default on).
2. Added `--no-dn-diff-detach-eta-cond` for gradient-through-eta experiments.
3. Applied in denoiser conditioning and control-path SNR mapping.

Primary references:

1. `train.py` CLI + constructor wiring.
2. `model.py` `dn_diff_predict` and control mapping paths.

## G. Frozen classifier mode consistency

1. Added consistent mode helper to keep frozen classifier stack in eval mode.
2. Aux trainable modules remain in train mode.

Primary references:

1. `model.py`: `set_frozen_classifier_train_mode`.
2. `train.py`: called each epoch when frozen mode active.

## H. Additional validation hardening

1. Beta schedule bounds validated (`0 < beta_start < beta_end < 1`).
2. Alignment losses require stable teacher (`frozen` or `ema`).
3. Frozen dn-diff runs require task-aware denoiser supervision (`lambda_dn_cls` and/or align loss).

---

## 4) What Is Still Not Solved

## R1. No true clean target by default

Even with the new task-aware loss, the diffusion target itself is still noisy-reference reconstruction unless a cleaner supervisory source is introduced.

## R2. Full elimination of duplicate pass is conditional

Cache reuse removes decoupling in the common path, but fallback recompute remains for incompatible cache conditions.

## R3. Low-band-only mask remains a deliberate tradeoff

If `dn_diff_apply_lowband_only_train=true`, training remains focused on selected SNR range by design.

---

## 5) Assessment of the 4 Forward Options (Ranked)

Scoring dimensions:

1. Expected impact on low-SNR classification
2. Implementation risk
3. Time-to-signal
4. Interpretability

### Option 1: Use diffusion loss as short warmup pretext, then down-weight hard

Recommendation: Keep  
Confidence: 0.84

Why:

1. Reduces identity-ceiling dominance from pure reconstruction objective.
2. Stabilizes early denoiser training without overcommitting to noisy-target reconstruction.

Risk:

1. If down-weight schedule is too aggressive, denoiser may underfit.

### Option 2: Make task-aware denoiser loss (`lambda_dn_cls`) primary

Recommendation: Strongly keep (primary lever)  
Confidence: 0.90

Why:

1. Directly optimizes denoised outputs for classifier objective.
2. Best practical fix under no true-clean-target constraint.
3. Already implemented and measurable via `train_loss_dn_cls` + val/test bands.

Risk:

1. Can overfit if too strong too early; needs small-to-moderate ramp.

### Option 3: Match conditioning distribution (`dn_diff_loss_cond_source=raw`)

Recommendation: Keep as default baseline  
Confidence: 0.86

Why:

1. Removes avoidable train/eval shift.
2. Improves interpretability of denoiser behavior.

Risk:

1. Might reduce robustness to larger corruption unless later reintroduced via controlled ablation.

### Option 4: Introduce cleaner targets (synthetic paired clean/noisy or pseudo-clean teacher)

Recommendation: Highest upside, medium-term milestone  
Confidence: 0.74

Why:

1. Only direct way to truly break noisy-target identity ceiling at objective level.
2. Most principled fix for denoiser semantics.

Risk:

1. Requires nontrivial data/teacher pipeline design.
2. Higher implementation complexity and confound risk.

---

## 6) Recommended Execution Order

1. Run next E.1/E.2 with:
   1. `dn_diff_loss_cond_source=raw`
   2. nonzero `lambda_dn_cls`
   3. one-step eval parity (DDIM mismatch override off)
2. Keep `lambda_dn_diff` as warmup regularizer, not dominant late objective.
3. Use alignment losses only after baseline task-aware run is stable.
4. If gains remain capped, start clean-target track (Option 4).

---

## 7) Key New/Updated Controls (Quick Reference)

1. `--lambda-dn-cls`
2. `--dn-diff-loss-cond-source {raw,degraded}`
3. `--dn-diff-allow-eval-ddim-mismatch`
4. `--dn-diff-require-noise-supervision`
5. `--dn-diff-detach-eta-cond` / `--no-dn-diff-detach-eta-cond`
6. `--dn-diff-train-forward-mode {raw,onestep}`

---

## 8) Verification Status

1. Static compile check passed:
   1. `python -m py_compile train.py model.py diffusion.py`
2. Runtime validation still required on training hosts (`torch` unavailable in this local shell environment).

