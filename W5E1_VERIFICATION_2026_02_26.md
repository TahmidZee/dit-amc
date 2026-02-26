# Wave 5E.1 Verification Report (2026-02-26)

## Scope

Full code review of the waveform diffusion denoiser implementation (Wave 5E.1), verification of incident report root causes, and identification of an additional design flaw.

## 1. Diffusion Model Approach — Verified Sound

The waveform diffusion denoiser front-end is conceptually correct and properly implemented:

- **V-prediction target**: The formulas `predict_x0_from_v`, `predict_eps_from_v`, and the training target `v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0` are mathematically correct (`diffusion.py:72-80`).
- **DDIM sampling**: `ddim_step()` and `ddim_sample_loop()` implement the standard DDIM reverse process correctly, including eta-controlled stochasticity and a same-step guard (`diffusion.py:82-153`).
- **Frozen-classifier isolation**: `set_dn_diff_train_freeze()` freezes all parameters except `dn_diff_model` and `noise_fraction_net` (`model.py:1852-1867`). Classification loss is zeroed via `cls_loss_mult = 0.0` (`train.py:3924`).
- **Conditional denoiser**: `WaveformDiffusionDenoiser1D` is a FiLM-conditioned U-Net with 3-level encoder/decoder, timestep + eta conditioning, and skip connections (`model.py:260-336`).
- **DDIM eval trajectory**: 8 steps from t_start→0 gives approximately uniform spacing (e.g., 85→74→63→53→42→31→21→10→0).
- **Forward process**: `q_sample` correctly implements `x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*eps` (`diffusion.py:63-65`).

## 2. Incident Report Root Causes — All Confirmed

### RC1: Frozen classifier without warm-start → chance-level (0.0909)

**Confirmed.** With 11 classes, 1/11 = 0.0909. A frozen random classifier will never improve since `cls_loss_mult = 0.0`.

**Fix verified**: `train.py:3334-3348` raises `ValueError` if frozen-classifier mode lacks `--init-ckpt`.

### RC2: EMA/warm-start mismatch

**Confirmed.** Original `--init-ckpt` loaded only `ckpt["model"]`, but eval could read the stale EMA shadow.

**Fix verified**: After loading `--init-ckpt`, the code reinitializes EMA via `EMA.create(model, decay)` (`train.py:3434-3436`), ensuring EMA shadow matches loaded weights.

### RC3: Early val decline is expected with frozen classifier

**Confirmed.** Smoke tests show both E0 (control) and E1 (diffusion) declining slightly over 2 epochs. This is natural: auxiliary losses update shared CNN trunk features, shifting them away from the frozen classifier's decision boundary. The E0 starting val_acc of 0.6239 is the correct E.1 baseline — it should not be compared to the historical ~0.64 from full supervised training, since E.1 is a frozen-classifier diagnostic.

## 3. New Finding: SNR-to-Timestep Schedule Saturation

### The problem

The `DiffusionSchedule` was created with **100 timesteps, linear beta from 1e-4 to 2e-2** (`model.py:1609`). This gives:

```
alpha_bar range: ~1.0 (t=0) to ~0.37 (t=99)
Minimum alpha_bar (0.37) corresponds to SNR ≈ -2.3 dB
```

The `snr_to_t()` mapping (`diffusion.py:155-161`) converts channel SNR to a diffusion timestep by finding the timestep whose `alpha_bar` matches `SNR_lin / (SNR_lin + 1)`:

| SNR (dB) | alpha_bar target | Maps to |
|---|---|---|
| +18 | 0.984 | t ≈ 2 |
| 0 | 0.500 | t ≈ 65 |
| -2.3 | 0.370 | t = 99 (schedule limit) |
| -6 | 0.200 | **clamped to t=99** |
| -14 | 0.038 | **clamped to t=99** |
| -20 | 0.010 | **clamped to t=99** |

**The entire target band [-14, -6] dB collapses to a single timestep (t=99).** The denoiser cannot distinguish -14 dB from -6 dB noise levels because the schedule doesn't go noisy enough.

### Evidence from smoke tests

- Training `dn_diff_t_start_mean = 85.37` with `std = 14.43` — heavily concentrated in the top portion of the schedule
- The mean is 85 (not 99) because training first degrades SNR by 2-8 dB, and the noise fraction net may overestimate SNR for severely degraded signals. But the underlying saturation remains.

### Code fix applied

Added CLI flags `--dn-diff-beta-start` and `--dn-diff-beta-end` (`train.py`, `model.py`) so the beta schedule endpoints can be configured without changing code. The `DiffusionSchedule` constructor already accepted these parameters; they just weren't exposed.

Added a schedule-coverage diagnostic printed at startup that computes `alpha_bar_min` and the minimum SNR covered, with a warning if the configured loss range is outside schedule coverage.

Added `dn_diff_beta_start`, `dn_diff_beta_end`, and `dn_diff_alpha_bar_min` to `metrics.jsonl` for tracking.

### Recommended settings

Two equivalent fixes (use either, not both):

**Option A — increase beta_end (keep 100 timesteps):**
```
--dn-diff-beta-end 0.10
```
This gives alpha_bar_min ≈ 0.007, covering SNR down to approximately -22 dB.

**Option B — increase timesteps (keep default beta_end):**
```
--dn-diff-train-timesteps 500
```
This gives alpha_bar_min ≈ 0.007, same coverage.

Option A is preferred since it keeps DDIM eval fast (fewer implicit steps in the schedule) without needing to increase `--dn-diff-eval-steps`.

## 4. Other Observations

### Noise fraction net is safely static in E.1 (no drift)

When `--dn-diff-enable` is true, `lambda_snr_eff` and `lambda_noise_eff` are hardcoded to 0.0 (`train.py:3925-3926`). The noise fraction net is unfrozen but all gradient paths are detached (`eta_est_dn.detach()` at `train.py:5160`, `eta_pred.detach()` at `model.py:1724`). Since `p.grad` stays `None`, AdamW skips these parameters entirely — no momentum update and no weight decay applied. The noise fraction net weights are effectively static at warm-start values throughout training. This is safe.

Warm-start calibration quality: `eta_pearson = 0.854, eta_spearman = 0.824` (from E0 smoke metrics).

### Training/eval active fraction mismatch

- Training loss mask: [-14, -6] dB only (`--dn-diff-apply-lowband-only-train`)
- Eval active range: signals below approximately -4 dB (`dn_diff_low_snr_thresh + dn_diff_high_snr_margin = -6 + 2 = -4`)

The denoiser is evaluated on [-20, -4] dB but only trained on [-14, -6] dB. Behavior on [-20, -14] and [-6, -4] dB is untrained extrapolation. Monitor per-SNR accuracy at eval to check for degradation at these edges.

## 5. Code Changes Summary

| File | Change |
|---|---|
| `train.py` | Added `--dn-diff-beta-start`, `--dn-diff-beta-end` CLI args |
| `train.py` | Pass beta values to all 3 CLDNNAMC construction sites |
| `train.py` | Added schedule-coverage diagnostic print at startup with warning |
| `train.py` | Added `dn_diff_beta_start`, `dn_diff_beta_end`, `dn_diff_alpha_bar_min` to metrics.jsonl |
| `model.py` | Added `dn_diff_beta_start`, `dn_diff_beta_end` constructor parameters |
| `model.py` | Pass beta values to `DiffusionSchedule` constructor |

All changes are backward-compatible: default values match previous hardcoded behavior.
