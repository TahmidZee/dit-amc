# Wave 5E.1 Incident Report (2026-02-26)

## Scope
This note summarizes what happened in Wave 5E.1 execution, why the first runs failed, what was fixed, and what to run next.

## Executive Summary
- The early `val_acc ~= 0.0909` failure was a configuration/wiring failure, not a diffusion-capability verdict.
- The primary failures were:
  1. Frozen classifier without pretrained init (first E.1 runs).
  2. Eval-time EMA mismatch during warm-start (`--init-ckpt` loaded model weights, while eval used EMA shadow).
- Smoke reruns after partial fixes recovered non-chance validation (`~0.62` control, `~0.585` diffusion core), proving the chance-collapse bug path is fixed.
- Additional correction now implemented: warm-start can use checkpoint EMA weights in frozen-classifier waves (auto mode).

## What We Observed

### Run behavior snapshot (Goose)
- `w5e1_ctrl_frozen_residual_s2016`
  - No `init_ckpt`, classifier frozen.
  - `best_val=0.0909` (chance), stayed flat.
- `w5e1_diff_core_pred_evalpred_s2016`
  - No `init_ckpt`, classifier frozen.
  - `best_val=0.0909` (chance), stayed flat.
- `w5e1_*_fix1`
  - Added `init_ckpt`.
  - Chance collapse removed, but eval remained unstable due to EMA/mode mismatch.
- `w5e1_smoke2_e0_ctrl_frozen_fix2` (2-epoch smoke)
  - `val_acc`: `0.6239 -> 0.6169`
- `w5e1_smoke2_e1_diff_core_fix2` (2-epoch smoke)
  - `val_acc`: `0.5851 -> 0.5847`

## Root Causes

### RC1: Frozen random classifier in initial E.1 runs
- In E.1 design, classifier is frozen.
- Without warm-start weights, chance-level validation is expected and permanent.

### RC2: Warm-start loaded `model` while eval read EMA shadow
- `--init-ckpt` was loading `ckpt["model"]` only.
- Eval path could use EMA shadow; EMA was not aligned to warm-start source in earlier behavior.
- This made early validation unreliable in frozen-classifier waves.

### RC3: E.1 objective naturally allows early val decline
- With classifier frozen and CE effectively removed in frozen mode, denoiser optimization can move features away from the frozen decision boundary.
- This can reduce val accuracy, especially in short runs.
- So a moderate early drop is not by itself proof of a fundamental architecture failure.

## Fixes Implemented in Code

### 1) EMA-safe warm-start behavior
- Reinitialize EMA shadow from loaded model after `--init-ckpt` load.
- Use EMA for evaluation only after `global_step >= ema_start`.

### 2) Warm-start source control
- Added CLI flag:
  - `--init-ckpt-source {auto,model,ema}`
- `auto` behavior:
  - For frozen-classifier waves, prefer checkpoint EMA state when available.
  - Otherwise load checkpoint model state.
- Logging added in `metrics.jsonl`:
  - `init_ckpt_source_used`

## Why Control Is Not Automatically `>0.64` in E.1
- The historical `~0.64` reference comes from full CLDNN training/eval flow.
- E.1 is a frozen-classifier denoiser feasibility setup, not full supervised classifier training.
- Therefore:
  - startup value depends on warm-start source quality,
  - and short-run denoiser updates can degrade a frozen classifier.

## Corrected E.1 Run Protocol
1. Always provide warm-start checkpoint for frozen-classifier runs.
2. Use `--init-ckpt-source auto` (or `ema`) for frozen-classifier E.1.
3. Keep EMA disabled in these diagnostics (`--ema-decay 0`) unless specifically testing EMA.
4. Do 1-2 epoch smoke runs first; abort if validation returns to chance level.

## Immediate Next Actions
1. Re-run E0/E1 smoke with new code and `--init-ckpt-source auto`.
2. If smoke is stable, launch full E.1 matrix.
3. Gate E.2 only if E.1 clears low-band feasibility gate.

