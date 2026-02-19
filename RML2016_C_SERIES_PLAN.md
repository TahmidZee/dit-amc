# RML2016 C-Series V2: ST-ACF + Low-SNR Raw-Path Gating (Aggressive 70% Plan)

## Date

2026-02-18

## Goal

- Primary objective: push RML2016.10a from ~0.640 to an aggressive 0.700 target.
- Main bottleneck: low-SNR band (`-14..-6 dB`) where current best is ~0.311.
- Execution policy: run **8 experiments at a time**, but run **multiple waves** (not limited to 8 total).

## What Is Implemented in Code (Current)

The following capabilities are available in `train.py` + `model.py`:

1. **Expert-v2 branch (new)**:
   - `--cldnn-expert-v2`
   - normalized local short-time correlations (Re/Im/|r_k|) with local power normalization
   - stable phase-diff features via dot/cross (no `atan2`)
   - tunable normalization epsilon: `--cldnn-expert-corr-eps`

2. **Eta-gated expert fusion (new)**:
   - `--cldnn-expert-eta-gate`
   - `--cldnn-expert-eta-gate-center`, `--cldnn-expert-eta-gate-tau`
   - `--cldnn-expert-eta-gate-min`, `--cldnn-expert-eta-gate-max`
   - gates expert maps/scalars down at high-SNR, up at low-SNR

3. **Cyclostationary scalar ablation switch**:
   - `--cldnn-cyclo-stats` (default on)
   - `--cldnn-no-cyclo-stats`

4. **Low-SNR raw-path attenuation in dual-path classifier**:
   - base controls: `--cldnn-raw-low-snr-drop-prob`, `--cldnn-raw-low-snr-drop-gate`, `--cldnn-raw-low-snr-drop-eta-thresh`, `--cldnn-raw-low-snr-drop-snr-thresh`, `--cldnn-raw-low-snr-drop-min-scale`, `--cldnn-raw-low-snr-drop-max-scale`
   - **SNR-shaped probability schedule (new)**:
     - `--cldnn-raw-low-snr-drop-prob-lo`, `--cldnn-raw-low-snr-drop-prob-mid`, `--cldnn-raw-low-snr-drop-prob-hi`
     - `--cldnn-raw-low-snr-drop-snr-lo`, `--cldnn-raw-low-snr-drop-snr-mid`

5. **Targeted L_feat controls (new)**:
   - source-band mask: `--lfeat-snr-lo`, `--lfeat-snr-hi`
   - degraded-band mask: `--lfeat-snr-new-lo`, `--lfeat-snr-new-hi`
   - log metric: `train_lfeat_active_frac` to verify L_feat is active in the intended band

6. **Constrained denoiser paired degradation controls (new)**:
   - `--dn-pair-delta-min`, `--dn-pair-delta-max`
   - `--dn-pair-snr-floor-db`
   - `--dn-pair-snr-new-lo`, `--dn-pair-snr-new-hi`
   - prevents over-hard pair generation (e.g., not forcing `-14 -> -20` unless requested)

7. **Low-SNR adaptive consistency deltas (new)**:
   - `--snr-consist-adaptive-delta`
   - `--snr-consist-low-snr-thresh`
   - `--snr-consist-low-delta-min`, `--snr-consist-low-delta-max`

8. Validation guards were added for invalid ranges and incompatible settings.

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

## Wave 3R (8 runs): Mixed Expert-v2 + Targeted Denoiser Training

Purpose: run **4 expert-v2 ablations** plus **4 non-expert low-SNR improvements** in one wave.

Wave-2 signals to exploit:
- consistency gave best overall trend
- soft SNR-gated attenuation gave best low-band trend
- L_feat was active but not targeted

| ID | Run name | Delta flags vs base | Expected effect |
|---|---|---|---|
| W3R-0 | `w3r_noexp_anchor_consist` | `--cldnn-raw-low-snr-drop-gate eta --cldnn-raw-low-snr-drop-prob 0.4 --cldnn-raw-low-snr-drop-eta-thresh 1.0 --cldnn-raw-low-snr-drop-min-scale 0.0 --cldnn-raw-low-snr-drop-max-scale 0.0 --snr-consist --snr-consist-lambda 0.5 --snr-consist-warmup 30 --snr-consist-ramp 10 --snr-consist-temp 2.0 --snr-consist-delta-min 2 --snr-consist-delta-max 6 --snr-consist-snr-lo -14 --snr-consist-snr-hi 6 --snr-consist-conf-thresh 0.3` | reproducible overall anchor |
| W3R-1 | `w3r_noexp_softsched` | `--cldnn-raw-low-snr-drop-gate snr --cldnn-raw-low-snr-drop-snr-thresh 18 --cldnn-raw-low-snr-drop-prob-lo 0.75 --cldnn-raw-low-snr-drop-prob-mid 0.45 --cldnn-raw-low-snr-drop-prob-hi 0.05 --cldnn-raw-low-snr-drop-snr-lo -10 --cldnn-raw-low-snr-drop-snr-mid -6 --cldnn-raw-low-snr-drop-min-scale 0.1 --cldnn-raw-low-snr-drop-max-scale 0.3` | stronger low-SNR shortcut suppression |
| W3R-2 | `w3r_noexp_softsched_consist_adapt` | W3R-1 + `--snr-consist --snr-consist-lambda 0.5 --snr-consist-warmup 30 --snr-consist-ramp 10 --snr-consist-temp 2.0 --snr-consist-delta-min 2 --snr-consist-delta-max 6 --snr-consist-snr-lo -14 --snr-consist-snr-hi -4 --snr-consist-snr-new-lo -16 --snr-consist-snr-new-hi -6 --snr-consist-conf-thresh 0.3 --snr-consist-adaptive-delta --snr-consist-low-snr-thresh -6 --snr-consist-low-delta-min 2 --snr-consist-low-delta-max 4` | low-band-biased consistency |
| W3R-3 | `w3r_noexp_softsched_lfeat_target` | W3R-1 + `--lambda-feat 0.03 --feat-ramp-epochs 5 --feat-encoder-ckpt ./runs/rml2016_athena/b0_a3_baseline/best.pt --dn-pair-delta-min 2 --dn-pair-delta-max 6 --dn-pair-snr-floor-db -14 --dn-pair-snr-new-lo -14 --dn-pair-snr-new-hi -6 --lfeat-snr-lo -14 --lfeat-snr-hi -6 --lfeat-snr-new-lo -14 --lfeat-snr-new-hi -6` | targeted L_feat + constrained pairs |
| W3R-4 | `w3r_expv2_plain` | W3R-1 + `--cldnn-expert-features --cldnn-expert-v2` | expert-v2 baseline |
| W3R-5 | `w3r_expv2_eta_gate` | W3R-1 + `--cldnn-expert-features --cldnn-expert-v2 --cldnn-expert-eta-gate --cldnn-expert-eta-gate-center 0.8 --cldnn-expert-eta-gate-tau 0.7 --cldnn-expert-eta-gate-min 0.0 --cldnn-expert-eta-gate-max 1.0` | test eta-gated fusion |
| W3R-6 | `w3r_expv2_eta_gate_no_cyclo` | W3R-5 + `--cldnn-no-cyclo-stats` | check if global scalars hurt after gating |
| W3R-7 | `w3r_expv2_eta_gate_lfeat_consist` | W3R-6 + targeted L_feat block from W3R-3 + adaptive low-band consistency block from W3R-2 | full expert-v2 stack |

Wave-3R exit criterion:
- Promote top-2 by low-band first, then overall.
- Require high-band drop no worse than `-0.002` vs `b3_dn48`.

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
- by end of Wave 3R, low-band should be >= 0.340 and overall >= 0.650
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
