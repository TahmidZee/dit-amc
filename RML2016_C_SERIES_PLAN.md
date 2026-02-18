# RML2016 C-Series V2: ST-ACF + Low-SNR Raw-Path Gating (Aggressive 70% Plan)

## Date

2026-02-18

## Goal

- Primary objective: push RML2016.10a from ~0.640 to an aggressive 0.700 target.
- Main bottleneck: low-SNR band (`-14..-6 dB`) where current best is ~0.311.
- Execution policy: run **8 experiments at a time**, but run **multiple waves** (not limited to 8 total).

## What Was Just Implemented in Code

The following capabilities are now available in `train.py` + `model.py`:

1. **Short-Time ACF smoothing in expert branch** (physics-guided denoising before expert CNN):
   - `--cldnn-expert-stacf-win <int>`
   - `0/1` disables smoothing; odd windows recommended (`5`, `9`, `13`).
   - Smoothing is complex-safe: moving-average is applied separately to real/imag channels.

2. **Cyclostationary scalar ablation switch**:
   - `--cldnn-cyclo-stats` (default on)
   - `--cldnn-no-cyclo-stats` (disable late global scalars)

3. **Low-SNR raw-path attenuation in dual-path classifier** (prevents raw shortcut):
   - `--cldnn-raw-low-snr-drop-prob`
   - `--cldnn-raw-low-snr-drop-gate` (`auto|eta|snr`)
   - `--cldnn-raw-low-snr-drop-eta-thresh`
   - `--cldnn-raw-low-snr-drop-snr-thresh`
   - `--cldnn-raw-low-snr-drop-min-scale`
   - `--cldnn-raw-low-snr-drop-max-scale`
   - Applied only in train mode and dual-path denoiser mode.

4. Validation guards were added for invalid ranges and incompatible settings.

## Clarified Technical Position (from last discussion)

- The concern about instantaneous conjugate products at low SNR is valid; ST-ACF is the right corrective direction.
- But the prior expert branch was not useless: expert feature maps are fused pre-LSTM, so temporal dynamics were still usable.
- We are not removing cyclostats blindly; we will ablate with/without to avoid throwing away useful global priors.
- Denoiser already uses noise-conditioned FiLM internally; classifier FiLM (`--cldnn-noise-cond`/`--cldnn-snr-cond`) is treated as late-stage comparison, not default.

## Shared Base Recipe (Anchor)

All runs below start from this base unless deltas are explicitly listed:

```
--arch cldnn --dataset rml2016a
--cldnn-denoiser --cldnn-denoiser-dual-path
--cldnn-denoiser-base-ch 48 --cldnn-denoiser-dropout 0.05
--fit-noise-proxy-calibration
--stage-a-epochs 12 --stage-b-epochs 16
--stage-a-no-cls
--stage-b1-cls2dn-scale 0.0 --stage-b2-cls2dn-scale 0.1
--lambda-dn 0.3 --lambda-id 0.03
--lambda-noise 0.1 --lambda-snr 0.0
--dropout 0.15 --label-smoothing 0.02
--aug-phase --aug-shift
--amp --batch-size 512 --epochs 120
--lr 5e-4 --min-lr 1e-5 --warmup-steps 500 --lr-decay-start-epoch 15
--weight-decay 1e-4 --early-stop-patience 25
--train-per 600 --val-per 200
--snr-cap-max-db 18 --normalize rms
--seed 2016 --num-workers 4
```

---

## Wave 1 (8 runs): Expert Front-End Search

Purpose: determine whether ST-ACF fixes low-SNR expert branch behavior and whether late cyclostats help or hurt.

| ID | Run name | Delta flags vs base | Expected effect |
|---|---|---|---|
| W1-0 | `w1_dn48_control` | none | control anchor |
| W1-1 | `w1_expert_raw` | `--cldnn-expert-features` | baseline expert branch |
| W1-2 | `w1_expert_stacf5` | `--cldnn-expert-features --cldnn-expert-stacf-win 5` | light smoothing |
| W1-3 | `w1_expert_stacf9` | `--cldnn-expert-features --cldnn-expert-stacf-win 9` | medium smoothing |
| W1-4 | `w1_expert_stacf13` | `--cldnn-expert-features --cldnn-expert-stacf-win 13` | stronger smoothing |
| W1-5 | `w1_expert_stacf9_no_cyclo` | `--cldnn-expert-features --cldnn-expert-stacf-win 9 --cldnn-no-cyclo-stats` | test temporal maps without global scalars |
| W1-6 | `w1_expert_stacf13_no_cyclo` | `--cldnn-expert-features --cldnn-expert-stacf-win 13 --cldnn-no-cyclo-stats` | same, stronger smoothing |
| W1-7 | `w1_expert_stacf9_no_dn` | `--cldnn-expert-features --cldnn-expert-stacf-win 9` plus **disable** `--cldnn-denoiser --cldnn-denoiser-dual-path`, set `--lambda-dn 0 --lambda-id 0 --lambda-noise 0`, and run with `--stage-a-epochs 0 --stage-b-epochs 0` | deconfound expert benefit from denoiser routing |

Wave-1 exit criterion:
- Pick top-2 by low-band (`-14..-6`) first, then overall as tiebreaker.
- Reserve candidate if one slot frees early: `w1_expert_stacf9_impair_aug` (`--aug-cfo 0.005 --aug-gain 0.15`).

---

## Wave 2 (8 runs): Force Denoiser Usage (Raw-Path Gating)

Purpose: eliminate the dual-path raw shortcut and force classifier reliance on denoised view at low SNR.

Use `<W1_BEST_EXPERT_FLAGS>` from Wave 1 winner in all runs below.

| ID | Run name | Delta flags (plus `<W1_BEST_EXPERT_FLAGS>`) | Expected effect |
|---|---|---|---|
| W2-0 | `w2_best_expert_anchor` | none | wave-2 control |
| W2-1 | `w2_rawdrop_p02_eta08_hard` | `--cldnn-raw-low-snr-drop-prob 0.2 --cldnn-raw-low-snr-drop-eta-thresh 0.8 --cldnn-raw-low-snr-drop-min-scale 0.0 --cldnn-raw-low-snr-drop-max-scale 0.0` | mild gating |
| W2-2 | `w2_rawdrop_p04_eta10_hard` | `--cldnn-raw-low-snr-drop-prob 0.4 --cldnn-raw-low-snr-drop-eta-thresh 1.0 --cldnn-raw-low-snr-drop-min-scale 0.0 --cldnn-raw-low-snr-drop-max-scale 0.0` | stronger gating |
| W2-3 | `w2_rawdrop_p05_eta12_hard` | `--cldnn-raw-low-snr-drop-prob 0.5 --cldnn-raw-low-snr-drop-eta-thresh 1.2 --cldnn-raw-low-snr-drop-min-scale 0.0 --cldnn-raw-low-snr-drop-max-scale 0.0` | aggressive gating |
| W2-4 | `w2_rawdrop_p04_snr-10_hard` | `--cldnn-raw-low-snr-drop-prob 0.4 --cldnn-raw-low-snr-drop-gate snr --cldnn-raw-low-snr-drop-snr-thresh -10 --cldnn-raw-low-snr-drop-min-scale 0.0 --cldnn-raw-low-snr-drop-max-scale 0.0` | isolate gating effect from eta calibration |
| W2-5 | `w2_rawdrop_lfeat` | W2-2 + `--lambda-feat 0.02 --feat-ramp-epochs 5 --feat-encoder-ckpt ./runs/rml2016_athena/b0_a3_baseline/best.pt` | preserve details in denoised path |
| W2-6 | `w2_rawdrop_consist` | W2-2 + `--snr-consist --snr-consist-lambda 0.5 --snr-consist-warmup 30 --snr-consist-ramp 10 --snr-consist-temp 2.0 --snr-consist-delta-min 2 --snr-consist-delta-max 6 --snr-consist-snr-lo -14 --snr-consist-snr-hi 6 --snr-consist-conf-thresh 0.3` | robustness focused on transition/low band |
| W2-7 | `w2_rawdrop_lfeat_consist` | W2-2 + both W2-5 and W2-6 deltas | strongest low-SNR stack candidate |

Wave-2 exit criterion:
- Choose one default candidate for Wave 3 based on:
  1) low-band gain, 2) no high-band collapse, 3) stable convergence.
- Reserve candidate if one slot frees early: soft eta-gating (`p=0.4`, `eta=1.0`, scale range `0..0.2`).

---

## Wave 3 (8 runs): Full Aggressive Stack + FiLM Head-to-Head + Seed Robustness

Purpose: maximize performance while testing whether classifier FiLM adds anything once denoiser + gating are strong.

Use `<W2_BEST_FLAGS>` from Wave 2 winner in all runs below.

| ID | Run name | Delta flags (plus `<W2_BEST_FLAGS>`) | Expected effect |
|---|---|---|---|
| W3-0 | `w3_best_anchor` | none | wave-3 control |
| W3-1 | `w3_reg_mixup` | `--mixup-alpha 0.2 --mixup-prob 0.3 --mixup-snr-min 6 --mixup-cls-only --dropout 0.20 --aug-cfo 0.005 --aug-gain 0.15` | stronger regularization |
| W3-2 | `w3_reg_mixup_snrbal` | W3-1 + `--snr-balanced --snr-balance-power 0.5` | low-SNR gradient reweighting |
| W3-3 | `w3_reg_mixup_snrbal_supcon` | W3-2 + `--supcon --supcon-lambda 0.1 --supcon-temp 0.07 --supcon-proj-dim 128 --supcon-warmup 30` | representation robustness |
| W3-4 | `w3_noise_film_head` | W3-3 + `--cldnn-noise-cond` | classifier eta-FiLM comparison |
| W3-5 | `w3_snr_film_head` | W3-3 + `--cldnn-snr-cond` | classifier SNR-FiLM comparison |
| W3-6 | `w3_best_seed2017` | best of W3-0..W3-5 + `--seed 2017` | seed robustness |
| W3-7 | `w3_best_seed2018` | best of W3-0..W3-5 + `--seed 2018` | seed robustness |

Wave-3 exit criterion:
- Top candidate must beat b3 on overall and low-SNR with acceptable high-SNR retention.

---

## Metrics and Decision Gates

Primary metrics:
- overall test mean over all SNRs
- low band mean (`-14..-6 dB`)
- mid band mean (`-4..+6 dB`)
- high band mean (`+10..+18 dB`)

Promotion gate vs `b3_dn48`:
- high-band drop no worse than -0.002 absolute
- and **at least one** of:
  - overall >= +0.006 absolute
  - low-band >= +0.015 absolute

Aggressive gate for 70% trajectory:
- by end of Wave 3, low-band should be >= 0.340 and overall >= 0.650
- if not achieved, consider architecture changes (sequence length, stronger denoiser depth, or pretext training), not just regularization tweaks.

---

## Early-Kill / Resource Policy

Since we can run 8 at once and many waves:

- run each wave in parallel (8 jobs)
- checkpoint monitor epochs: 40, 60, 80
- kill a run if both hold by epoch 60:
  - best val < (current wave median - 0.004)
  - low-band val trend is flat/down for >= 12 epochs

Freed slots should be used immediately for next-wave candidates.

---

## Command Strategy

Use one shared base command and append per-run delta flags from the tables above.
This keeps reproducibility high and avoids command drift across machines.

Machine path reminder:
- Goose: `/home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC`
- Athena: `/lustre/home/tahit/Modulation/dit-amc`

---

## Why This Plan Is Comprehensive

- It directly addresses the two core concerns from the last message:
  - local expectation in expert branch via ST-ACF
  - forcing denoiser-path usage via low-SNR raw-path attenuation
- It keeps strict ablation structure (single mechanism -> interactions -> full stack).
- It includes invariance-focused augmentation and conditioning head comparisons.
- It explicitly scales beyond 8 experiments while respecting 8-concurrent capacity.
