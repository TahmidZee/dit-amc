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

8. **External KD + low-band KD controls (new)**:
   - external teacher: `--teacher-ckpt`, `--lambda-kd`, `--kd-temp`
   - schedule: `--kd-warmup`, `--kd-ramp`
   - low-band KD gating: `--kd-snr-lo`, `--kd-snr-hi`
   - optional high-band preservation KD (two-mask): `--kd-hi-preserve-scale`, `--kd-hi-snr-lo`, `--kd-hi-snr-hi`, `--kd-hi-conf-thresh`
   - confidence gate: `--kd-conf-thresh`
   - teacher conditioning mode: `--kd-teacher-snr-mode` (default `known` for oracle-teacher distillation)
   - log metrics: `train_loss_kd`, `train_kd_active_frac`, `train_loss_kd_hi`, `train_kd_hi_active_frac`

9. Validation guards were added for invalid ranges and incompatible settings.
10. Eval reports now include class-macro metrics:
   - `val_macro_acc`, `val_macro_f1`, `val_low_macro_acc`, `val_low_macro_f1`
   - test summary file: `test_macro_summary.json`

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

## Wave-3R Outcome (2026-02-19 pull)

Observed test metrics (overall / low / mid / high):
- `w3r_noexp_anchor_consist`: `0.6408 / 0.3108 / 0.8935 / 0.9351` (best overall in W3R)
- `w3r_noexp_softsched`: `0.6399 / 0.3124 / 0.8885 / 0.9361` (best balanced low/high in W3R)
- `w3r_expv2_eta_gate_lfeat_consist`: `0.6373 / 0.3172 / 0.8794 / 0.9319` (best low-band, but high-band drops too much)
- all other W3R runs are below the above fronts on either overall or low-band.

Decision:
- No run met the promotion gate (`overall +0.006` or `low +0.015` vs `b3_dn48` with high-band guard).
- Move forward with **KD-first** strategy (Wave-4D), while keeping one non-KD anchor.
- Expert-v2 remains secondary unless high-band protection is improved.

---

## Wave 4T (Prerequisite): Build Oracle Teachers (2 runs)

Need:
- There is currently no `snr_mode=known` checkpoint in the run pool, so train at least one oracle teacher first.

| ID | Run name | Delta flags vs best non-expert W3R recipe | Purpose |
|---|---|---|---|
| W4T-0 | `w4t_teacher_oracle_anchor` | `--snr-mode known` | primary teacher |
| W4T-1 | `w4t_teacher_oracle_anchor_seed3407` | `--snr-mode known --seed 3407` | seed-robust teacher backup |

Teacher selection rule:
- choose teacher with best test overall, tie-break by low-band.

---

## Wave 4D (Activated): Oracle-Teacher Distillation (LUPI)

Trigger condition:
- Activated (Wave-3R remained in micro-gain territory).

Core idea:
- Train/keep an **oracle teacher** (`snr_mode=known`) as privileged training information.
- Train deployment student in **blind mode** (`snr_mode=predict`) with CE + KD (+ optional consistency/L_feat).
- This is training-time privilege only; inference remains blind.

Scenario handling:
- If teacher and student architectures differ, KD is allowed but lower priority than same-arch KD.
- Start with same-arch teacher first for clean attribution.
- Keep low-band KD gating enabled (`--kd-snr-lo -14 --kd-snr-hi -6`) to focus the distillation budget where it matters.

Wave-4D ablation base (fixed):
- Use `w3r_noexp_softsched` as the baseline recipe (best low/high balance in W3R without expert branch).
- Add KD deltas on top of this fixed base to keep attribution clean.
- Prevent high-SNR leakage in raw attenuation schedule:
  - set `--cldnn-raw-low-snr-drop-prob-hi 0.0` (or use `--cldnn-raw-low-snr-drop-zero-hi`)

Wave-4D matrix (8 student runs):

| ID | Run name | Delta flags vs promoted W3R baseline | Expected effect |
|---|---|---|---|
| W4D-0 | `w4d_student_anchor` | no KD | baseline for attribution |
| W4D-1 | `w4d_kd_fullband_l02_t2` | `--teacher-ckpt <oracle_teacher.pt> --lambda-kd 0.2 --kd-temp 2.0 --kd-warmup 20 --kd-ramp 10` | broad KD regularization |
| W4D-2 | `w4d_kd_lowband_l02_t2` | W4D-1 + `--kd-snr-lo -14 --kd-snr-hi -6` | focused low-band transfer |
| W4D-3 | `w4d_kd_lowband_l03_t2` | W4D-2 + `--lambda-kd 0.3` | stronger low-band KD |
| W4D-4 | `w4d_kd_lowband_l02_t3` | W4D-2 + `--kd-temp 3.0` | softer teacher targets |
| W4D-5 | `w4d_kd_twomask_l025_h005` | W4D-2 + `--lambda-kd 0.25 --kd-hi-preserve-scale 0.20 --kd-hi-snr-lo 10 --kd-hi-snr-hi 18` | low-band KD + explicit high-band drift control |
| W4D-6 | `w4d_kd_lowband_consist` | W4D-2 + low-band consistency block from W3R-2 | KD + invariance |
| W4D-7 | `w4d_kd_lowband_lfeat` | W4D-2 + targeted L_feat block from W3R-3 | KD + denoiser detail preservation |

Wave-4D exit criterion:
- Promote if KD run beats W3R best on low-band with non-collapsing high-band.
- Prefer runs with meaningful KD activity (`train_kd_active_frac`) in target band and stable convergence.

---

## Wave 4D-Fix (Activated Next): KD Stabilization + Privileged Transfer

Reason:
- Wave-4D kept the same Pareto shape (low-band up, mid/high down), which is consistent with KD over-steering a shared trunk.
- Feedback-2 fixes were implemented in `train.py` before moving to MoE.

Implemented KD fixes in code:
- KD scaling now defaults to **batch-mean** (not active-count mean) to avoid low-band over-amplification.
- **Correctness-filtered KD** added (`--kd-correctness-filter`, default on).
- Stage-aware KD start added:
  - `--kd-warmup-after-stages` (default on)
  - `--kd-post-stage-delay 8` (default), so KD starts after Stage-A/B + delay.
- Mixup conflict control:
  - `--kd-disable-mixup` to run KD without mixup as a clean sanity condition.
- New optional privileged-transfer channels:
  - **Denoiser KD**: `--lambda-kd-denoise`, low-band gated by `--kd-denoise-snr-lo/hi`
  - **Pre-FiLM feature KD**: `--lambda-kd-feat`, low-band gated by `--kd-feat-snr-lo/hi`

Wave-4D-Fix matrix (8 runs):

| ID | Run name | Delta flags vs `w3r_noexp_softsched` base | Expected effect |
|---|---|---|---|
| W4DF-0 | `w4df_kd_low_l020_stable` | low-band KD + `--kd-disable-mixup --kd-correctness-filter --kd-warmup-after-stages --kd-post-stage-delay 8` | sanity-stable baseline |
| W4DF-1 | `w4df_kd_low_l020_twomask_stable` | W4DF-0 + high-band preserve KD (`--kd-hi-preserve-scale 0.20`) | test high-band retention after scaling fix |
| W4DF-2 | `w4df_kd_low_l025_twomask` | W4DF-1 + `--lambda-kd 0.25` | stronger KD under stabilized scaling |
| W4DF-3 | `w4df_kd_low_l020_t30` | W4DF-0 + `--kd-temp 3.0` | softer teacher targets |
| W4DF-4 | `w4df_kd_low_l020_dnkd_l010` | W4DF-0 + `--lambda-kd-denoise 0.10` | direct oracle->blind denoiser transfer |
| W4DF-5 | `w4df_kd_low_l020_featkd_l010` | W4DF-0 + `--lambda-kd-feat 0.10` | transfer representation geometry (pre-FiLM) |
| W4DF-6 | `w4df_kd_low_l020_dnfeat` | W4DF-0 + both denoiser KD + feature KD (`0.08/0.08`) | combined privileged transfer |
| W4DF-7 | `w4df_kd_low_l020_dnfeat_twomask` | W4DF-6 + `--kd-hi-preserve-scale 0.20` | full stabilized KD stack + high guard |

Wave-4D-Fix decision gate:
- Must beat W3R low-band while keeping high-band drop <= `0.002` absolute.
- If two-mask still cannot remove mid/high tax, promote MoE immediately.

---

## Wave 4D-Fix Outcome (2026-02-22 pull)

Observed validation-only metrics (test not yet finalized; runs at 62–96 epochs):

| Run | Val overall | Val low | Val mid | Val high | Δ low vs softsched |
|---|---|---|---|---|---|
| `w4df2_stable_lowkd` | 0.6386 | 0.3070 | 0.8898 | 0.9343 | +0.0013 |
| `w4df2_stable_lowkd_highpres` | 0.6389 | 0.3112 | 0.8878 | 0.9339 | +0.0055 |
| `w4df2_stable_lowkd_midpres` | 0.6391 | 0.3101 | 0.8891 | 0.9339 | +0.0044 |
| `w4df2_stable_lowkd_dnkd` | 0.6370 | 0.3067 | 0.8873 | 0.9323 | +0.0010 |
| `w4df2_stable_lowkd_featkd` | 0.6376 | 0.3046 | 0.8902 | 0.9335 | −0.0011 |
| `w4df2_stable_lowkd_dnfeat` | 0.6367 | 0.3049 | 0.8886 | 0.9326 | −0.0008 |
| `w4df2_stable_lowkd_dnfeat_midpres` | — | — | — | — | (0 epochs) |

Decision:
- KD closed: the low/high Pareto tradeoff persists. Best KD runs gain +0.005 low-band but lose −0.004 mid-band.
- Oracle ceiling audit revealed the **architecture itself** caps performance at 0.6855 (even with perfect SNR). The single-head 128-dim unidirectional LSTM cannot reach 0.70.
- **Architecture scaling is mandatory.** Proceed to expanded Wave 5.

---

## Architecture Ceiling Analysis (justifies Wave 5 redesign)

| Model | Overall | Low (−20…−6) | Mid (−4…+6) | High (+8…+18) |
|---|---|---|---|---|
| Best blind (`w3r_anchor_consist`) | 0.6408 | 0.2303 | 0.8935 | 0.9355 |
| Best KD blind (`w4d2_twomask`) | 0.6386 | 0.3220 | 0.8793 | 0.9335 |
| Oracle teacher (true SNR, same arch) | **0.6855** | 0.3239 | 0.9111 | 0.9419 |
| **Target** | **0.7000** | ≥0.35 | ≥0.93 | ≥0.95 |

Current architecture bottlenecks:
1. LSTM: **unidirectional**, 128 hidden → only 128-dim temporal features
2. Classifier head: `fc1(128→128) → fc2(128→128) → fc_out(128→11)` = **34K params** for 11 classes
3. Denoiser = 789K (65% of model) but classifier = 34K (2.8%) — capacity mismatch
4. No mechanism to decouple low-SNR vs high-SNR decision boundaries

The 0.70 target requires **low ≥ 0.35, mid ≥ 0.93, high ≥ 0.95** (all three must improve).

---

## Wave 5A (8 runs): Architecture Scaling Foundation

Purpose: raise the architecture ceiling while **keeping the CNN+LSTM family** (no backbone swap yet); establish strong scaled baselines before adding MoE.

Key changes (already supported):
- `--cldnn-bidir`: bidirectional temporal modeling (doubles LSTM output width)
- `--cldnn-lstm-layers 3`: deeper temporal stack (baseline uses 2 layers)
- `--cldnn-cls-hidden 384/512`: wider classifier head
- targeted dropout ablations (`--dropout 0.30`) on deeper/wider variants to guard against high-SNR overfit

Base recipe: `w3r_noexp_softsched` recipe (best low/high balance) with same denoiser, staging, and augmentation.

| ID | Run name | Delta flags vs softsched base | Expected effect |
|---|---|---|---|
| W5A-0 | `w5a_bidir2_cls256` | `--cldnn-bidir --cldnn-lstm-layers 2 --cldnn-cls-hidden 256` | scaled control |
| W5A-1 | `w5a_bidir2_cls384` | `--cldnn-bidir --cldnn-lstm-layers 2 --cldnn-cls-hidden 384` | width scaling (2-layer) |
| W5A-2 | `w5a_bidir2_cls512` | `--cldnn-bidir --cldnn-lstm-layers 2 --cldnn-cls-hidden 512` | stronger width scaling (2-layer) |
| W5A-3 | `w5a_bidir3_cls384` | `--cldnn-bidir --cldnn-lstm-layers 3 --cldnn-cls-hidden 384` | deeper temporal modeling + wider head |
| W5A-4 | `w5a_bidir3_cls512` | `--cldnn-bidir --cldnn-lstm-layers 3 --cldnn-cls-hidden 512` | max blind-capacity candidate |
| W5A-5 | `w5a_bidir3_cls384_do30` | W5A-3 + `--dropout 0.30` | overfit guard for deeper model |
| W5A-6 | `w5a_bidir3_cls512_do30` | W5A-4 + `--dropout 0.30` | overfit guard for max blind-capacity model |
| W5A-7 | `w5a_oracle_bidir3_cls512` | W5A-4 + `--snr-mode known` | **oracle ceiling** on max-capacity CNN+LSTM |

Wave-5A exit criterion:
- W5A-7 oracle must improve by **≥ +0.015** over previous oracle (~0.6855) **OR** exceed 0.70.
  - Rationale: a hard 0.70 gate is brittle; if oracle reaches 0.695, it still proves scaling helps and MoE/TTA/ensemble can push it over.
- Pick best blind run from W5A-0 through W5A-6 as new anchor.
- **If W5A-7 oracle fails both conditions**: stay in CNN+LSTM and scale further (`--cldnn-lstm-layers 4`, `--cldnn-lstm-hidden 160/192`, and/or `--cldnn-merge-ch 128/160`) before considering a backbone swap.

---

## Wave 5A Outcome (Goose + Athena pull, 2026-02-22)

Observed best test metrics by run family (overall / low `-20..-6` / mid / high):
- `w5a_bidir2_cls512`: `0.6409 / 0.2311 / 0.8917 / 0.9365` (**best blind**)
- `w5a_bidir3_cls512`: `0.6390 / 0.2269 / 0.8907 / 0.9370`
- `w5a_bidir3_cls512_do30`: `0.6408 / 0.2296 / 0.8921 / 0.9377` (best deep variant, still flat vs 2-layer)
- `w5a_oracle_bidir3_cls512` (`snr_mode=known`): `0.6829 / 0.3201 / 0.9073 / 0.9423`

Key read:
- Deeper LSTM did **not** move the frontier in a meaningful way (mostly micro-tradeoffs).
- Width helped slightly (`cls512` > `cls256/384`), but gains remain sub-pp.
- Oracle did not raise the ceiling enough to justify more depth-first scaling.

Metric-definition note (important):
- Low-band mismatch was a reporting artifact in prior discussion:
  - `-20..-6` low-band is ~0.23 in both W3R and W5A.
  - `-14..-6` low-band is ~0.31 in both W3R and W5A.

Decision:
- Promote `w5a_bidir2_cls512` as Wave-5B anchor.
- Move to **MoE head decoupling** as primary next lever.

---

## Wave 5B (8 runs): Specialized MoE Heads + Oracle-Gate Diagnostic

Purpose: fix the observed low/high frontier by forcing expert specialization explicitly, not just via blended-logits CE.

Status:
- Prior W5B (blended CE + gate regularizers only) was flat/slightly down despite good gate behavior.
- New W5B adds head-specific supervision, gate anneal, anti-collapse warmup, and a one-shot oracle-gate diagnostic.
- This wave is **not** teacher-student distillation; it is purely MoE-head training.

Key new flags:
- `--moe-head-low-lambda`, `--moe-head-high-lambda`: auxiliary CE on low/high experts
- `--moe-head-low-snr-lo/hi`, `--moe-head-high-snr-lo/hi`: SNR masks for head-specific CE
- `--moe-gate-tau-start`, `--moe-gate-tau-anneal-epochs`: soft-to-sharp gate schedule
- `--moe-entropy-warmup-lambda`, `--moe-entropy-warmup-epochs`: early anti-collapse entropy max
- `--moe-specialize-start-epoch`: delay entropy-min specialization until gate is stable
- `--moe-diversity-lambda`: tiny anti-copy penalty between expert logits
- `--moe-oracle-gate-train`: diagnostic mode that routes with true SNR (training-time only)

Base recipe: `w5a_bidir2_cls512` (best blind from W5A).

Specialization masks (overlap by design):
- Low-head CE mask: `-14..+2 dB`
- High-head CE mask: `-6..+18 dB`

| ID | Run name | Delta flags vs W5A best | Expected effect |
|---|---|---|---|
| W5B2-0 | `w5b2_spec_a020_h020` | `--moe-n-experts 2 --moe-gate-type eta-sigmoid --moe-gate-center 1.55 --moe-gate-tau 0.30 --moe-balance-lambda 0.01 --moe-head-low-lambda 0.20 --moe-head-high-lambda 0.20 --moe-head-low-snr-lo -14 --moe-head-low-snr-hi 2 --moe-head-high-snr-lo -6 --moe-head-high-snr-hi 18` | specialization anchor |
| W5B2-1 | `w5b2_spec_a020_h020_tauanneal` | W5B2-0 + `--moe-gate-tau-start 0.70 --moe-gate-tau-anneal-epochs 40` | soft early routing, sharper late |
| W5B2-2 | `w5b2_spec_a030_h020_tauanneal` | W5B2-1 + `--moe-head-low-lambda 0.30` | stronger low-head pressure |
| W5B2-3 | `w5b2_spec_a020_h030_tauanneal` | W5B2-1 + `--moe-head-high-lambda 0.30` | stronger high-head protection |
| W5B2-4 | `w5b2_spec_a020_h020_entwarm` | W5B2-1 + `--moe-entropy-warmup-lambda 0.01 --moe-entropy-warmup-epochs 20` | anti-collapse guard during early epochs |
| W5B2-5 | `w5b2_spec_a020_h020_div001` | W5B2-1 + `--moe-diversity-lambda 0.001` | prevent near-identical heads |
| W5B2-6 | `w5b2_oraclegate_a020_h020` | W5B2-1 + `--moe-oracle-gate-train` | diagnostic: isolate gate-vs-trunk bottleneck |
| W5B2-7 | `w5b2_oraclegate_a020_h020_entwarm` | W5B2-6 + `--moe-entropy-warmup-lambda 0.01 --moe-entropy-warmup-epochs 20` | oracle diagnostic with anti-collapse warmup |

Wave-5B exit criterion:
- Eta-gated run must beat W5A best on overall by ≥ +0.005 and low-band by ≥ +0.010, with high-band drop no worse than −0.003.
- Oracle-gated diagnostic interpretation:
  - If oracle-gated MoE does not beat W5A by at least +0.005 overall, trunk bottleneck is likely dominant.
  - If oracle-gated beats W5A but eta-gated does not, routing/training (not trunk capacity) is the blocker.

### Wave 5B.3 Recovery Mini-Wave (deconfounded diagnostics)

Status update: W5B.2 showed strong overfit and did not isolate the root cause cleanly.  
W5B.3 adds deconfounded diagnostics with minimal API changes:

- `--moe-oracle-gate-eval`: allow true oracle routing during eval for clean D1 vs D2 interpretation.
- `--moe-head-ce-detach-trunk`: force head-specific CE to update expert heads only.
- `--moe-head-ce-source clean|cls`: choose whether aux CE uses clean forward logits or class-forward logits.
- `--moe-head-ce-warmup`, `--moe-head-ce-ramp`: stabilize specialization CE onset.
- Validation diagnostics now include:
  - per-expert standalone accuracy (overall + low band),
  - gate entropy,
  - gate loads overall and in transition band (`--moe-transition-snr-lo/hi`, default `-8..-2 dB`).

W5B.3 diagnostic gates (low-band first):

- D2 vs D1 (oracle eval routing): pass if low-band improves by ≥ +0.010 **or** overall by ≥ +0.003.
- D3 vs D2 (oracle + head-only CE): require no collapse and low-band gain ≥ +0.010 (overall gain may be smaller).
- Deployable gate (eta-routed): require overall ≥ +0.003 with low-band ≥ +0.010 and high-band drop ≤ 0.003.
- Pivot trigger: if all head-only candidates fail both low-band ≥ +0.010 and overall ≥ +0.003 under the high-band constraint, pivot to SSL/trunk-level changes.

---

## Wave 5B.4 (2 runs): MoE Nondetach Confirmation

Purpose: close the remaining ambiguity from W5B.3 (detach vs nondetach) with a minimal, same-host confirmation before committing to pivot.

Status from completed W5B.3 + Goose companion:
- `D2 vs D1` (oracle eval routing) passed strongly.
- `D3 vs D2` (head-only detached CE under oracle routing) failed.
- Deployable eta-gated head-CE runs failed gate vs `D0`.
- Goose nondetach companion improved over detached (`+0.0185` overall, `+0.0448` low on peak val), but still remained below `D2`.

Runs:

| ID | Run name | Delta flags | Purpose |
|---|---|---|---|
| C0 | `w5b4_oratraineval_headonly_a005_nodetach_athena` | W5B.3 oracle-head run with `a=0.05`, **without** `--moe-head-ce-detach-trunk`, with `--moe-oracle-gate-train --moe-oracle-gate-eval`, plus `--epochs 80 --early-stop-patience 12 --early-stop-min-delta 0.0005` | same-host nodetach oracle confirmation |
| C1 | `w5b4_eta_headonly_a005_nodetach_athena` | eta-gated counterpart of C0 (no oracle flags), same early-stop settings | deployable nodetach check |

Decision gates:
- `C0 vs D2`: pass if `overall >= +0.003` **or** `low(-14..-6) >= +0.010`.
- `C1 vs D0`: pass if `overall >= +0.003` and `low(-14..-6) >= +0.010` with `high(+6..+18) drop <= 0.003`.
- If `C1` fails: stop MoE specialization tuning and pivot to SSL/trunk-level work.

---

## Wave 5C-SSLv2 (8 runs): Aggressive MoCo-v2 + SupCon Redesign

Purpose: move SSL from legacy in-batch contrastive to a stronger representation-learning stack before committing to trunk swap.

Why this replaces prior W5C(6-run) plan:
- historical strong contrastive run was oracle-conditioned (`snr_mode=known`), not a fair blind baseline,
- prior SupCon runs underperformed stronger non-SupCon anchors,
- legacy NT-Xent path lacked momentum target encoder + large negative queue.

Implementation status (now available in code):
- `--moco-pretrain-epochs`, `--moco-temp`, `--moco-momentum`, `--moco-queue-size`, `--moco-proj-dim`, `--moco-hidden-dim`, `--moco-lr`, `--moco-weight-decay`
- `--ssl-aug-awgn-*`, `--ssl-aug-time-mask-*`, `--ssl-aug-iq-drop-prob`
- `--supcon-clean-branch-on-mixup-cls-only`
- pretrain metrics file: `ssl_pretrain.jsonl`
- epoch metrics fields: `train_loss_supcon`, `train_supcon_active_frac`, plus MoCo config fields

Run matrix:

| ID | Run name | Delta flags | Purpose |
|---|---|---|---|
| S0 | `w5c2_ssl_anchor_nomoe` | non-MoE anchor (`--moe-n-experts 1`) | baseline reference |
| S1 | `w5c2_ssl_legacy_ntxent20` | S0 + `--contrastive-pretrain-epochs 20 --contrastive-k 4 --contrastive-temp 0.10` | legacy NT-Xent comparator |
| S2 | `w5c2_ssl_moco20_base` | S0 + `--moco-pretrain-epochs 20 --moco-temp 0.20 --moco-momentum 0.999 --moco-queue-size 16384 --moco-proj-dim 128 --moco-hidden-dim 512` | MoCo baseline |
| S3 | `w5c2_ssl_moco30_temp015` | S2 but `--moco-pretrain-epochs 30 --moco-temp 0.15` | stronger MoCo |
| S4 | `w5c2_ssl_supcon_clean_l010` | S0 + `--supcon --supcon-lambda 0.10 --supcon-warmup 10 --supcon-proj-dim 128 --supcon-temp 0.07 --supcon-clean-branch-on-mixup-cls-only` | SupCon clean-branch |
| S5 | `w5c2_ssl_supcon_clean_l015` | S4 + `--supcon-lambda 0.15` | SupCon dose response |
| S6 | `w5c2_ssl_moco20_supcon_l010` | S2 + S4 flags | combined MoCo + SupCon |
| S7 | `w5c2_ssl_moco20_supcon_l010_strongaug` | S6 + `--ssl-aug-awgn-prob 0.5 --ssl-aug-awgn-snr-min-db 2 --ssl-aug-awgn-snr-max-db 14 --ssl-aug-time-mask-prob 0.4 --ssl-aug-time-mask-max-frac 0.12` | robustness stress |

Wave-5C-SSLv2 promotion gate:
- Best run must satisfy all:
  - `test_acc >= max(S0, 0.6409) + 0.008`
  - low-band (`-20..-6`) gain `>= +0.015` vs `S0`
  - high-band (`+6..+18`) drop `<= 0.003` vs `S0`

If SSLv2 fails gate:
- start trunk-swap implementation wave (TCN/dilated temporal backbone).

---

## Wave 5D (inference-time): Test-Time Augmentation + Ensemble

Purpose: squeeze final accuracy without retraining. These are **inference-only** changes.

Key new flags (eval mode only):
- `--tta-views N` (NEW): number of augmented views to average at test time
- `--tta-phase` (NEW): apply random phase rotation per view
- `--tta-shift` (NEW): apply circular time shift per view
- `--ensemble-ckpts PATH1,PATH2,...` (NEW): average logits from multiple checkpoints

Expected gains (conservative — plan for lower bound):
- TTA (5 views, phase + shift): **+0.005 to +0.015** overall (free at inference, just slower)
- Ensemble (top-3 seeds): **+0.005 to +0.015** on top of TTA
- Combined realistic range: single-model 0.66 → TTA ~0.67–0.675 → Ensemble ~0.68–0.69
- ⚠️ Treat Wave 5D as a "squeeze" stage, not a guaranteed +0.04. Validate each technique's marginal gain before stacking.

Evaluation runs (no training):
1. `w5d_tta5_phase_shift` — best W5C model with `--tta-views 5 --tta-phase --tta-shift`
2. `w5d_tta10_phase_shift` — 10-view TTA for upper bound
3. `w5d_ensemble_top3` — top-3 models from W5C (logit averaging)
4. `w5d_ensemble_top3_tta5` — ensemble + TTA combined

---

## Code Implementation Checklist (before Wave 5A)

### Already supported (no code change needed):
- `--cldnn-bidir` → bidirectional LSTM
- `--cldnn-lstm-layers N` → LSTM depth (baseline currently 2)
- `--cldnn-lstm-hidden N` → LSTM hidden size
- `--cldnn-cls-hidden N` → wider classifier head
- `--label-smoothing FLOAT` → label smoothing
- `--focal-gamma FLOAT` → focal loss strength (`0` = standard CE)

### Must implement for Wave 5A:
- None. Wave-5A architecture sweep uses existing CNN+LSTM scaling knobs only.

### Optional (defer to Wave 5C if architecture scaling stalls):
1. **Focal loss** (`--focal-gamma`):
   - `FL(p_t) = -(1-p_t)^gamma * log(p_t)`, with `gamma=0` equivalent to CE
   - ⚠️ **Do NOT stack with label smoothing initially.** Run focal with `--label-smoothing 0.0` first; if needed, reintroduce small smoothing (0.01–0.02) after validating focal alone.

2. **SNR-weighted CE** (`--snr-weight-ce`, `--snr-weight-ce-scale`, `--snr-weight-ce-max`):
   - Per-sample weight: `w_i = 1 + scale * max(0, (snr_max - snr_i) / (snr_max - snr_min))`
   - Low-SNR samples get higher CE weight
   - ⚠️ **Cap max weight** (default 3x): `w_i = min(w_i, snr_weight_ce_max)` to avoid destabilization
   - ⚠️ **Normalize weights** to keep batch mean weight ~= 1.0: `w_i = w_i / mean(w_batch)` so effective LR does not drift
   - Without cap + normalization, this can recreate the same "low up, mid/high down" tradeoff seen in KD

### Must implement for Wave 5B:
3. **MoE classifier head** (`--moe-n-experts`, `--moe-gate-type`, etc.):
   - `MoEClassifierHead` module: N parallel (fc1→fc2→fc_out) expert stacks
   - Router takes `eta_pred` (and optionally features) to produce blend weights
   - Gate types: `eta-sigmoid` (no learned params), `learned` (MLP), `hard` (argmax)
   - Output: weighted sum of expert logits **plus per-expert logits** for auxiliary specialization losses
   - Load-balancing loss: `L_bal = lambda * Var(expert_load)` where load = mean gate weight per expert
   - Specialization loss: `L_spec = lambda_spec * H(gate_distribution)` — **minimizing H drives routing to be peaked** (low entropy = expert specialization). Previous version had wrong sign.
   - Add **head-specific CE supervision** (core W5B.2 fix):
     - `L_low = α_low * CE(logits_low_head, y)` on mask `snr in [low_lo, low_hi]`
     - `L_high = α_high * CE(logits_high_head, y)` on mask `snr in [high_lo, high_hi]`
     - Use overlapping masks (`low: -14..+2`, `high: -6..+18`) and normalize each masked loss by active-mask count.
   - Add **gate tau annealing**:
     - start with softer gate (`--moe-gate-tau-start`) and anneal to base `--moe-gate-tau` over `--moe-gate-tau-anneal-epochs`
   - Add **early anti-collapse entropy warmup**:
     - maximize gate entropy early via `--moe-entropy-warmup-lambda`, then decay over `--moe-entropy-warmup-epochs`
   - Add delayed specialization start:
     - apply `--moe-specialize-lambda` only after `--moe-specialize-start-epoch`
   - Add optional tiny diversity penalty:
     - `--moe-diversity-lambda` on cosine similarity between expert logits
   - Add oracle-gate diagnostic mode:
     - `--moe-oracle-gate-train` routes using true SNR during training to isolate gate-vs-trunk failure mode.
   - ⚠️ **Do NOT enable large specialization/diversity weights early.** Keep regularizers tiny until per-expert load is stable.
   - ⚠️ **CRITICAL — Router Gradient Leakage Trap**: If `--moe-gate-type learned`, the router's `L_bal` and `L_spec` losses generate gradients. If those flow backward through `eta_pred` or the CNN trunk, they will **destroy NoiseFractionNet calibration and feature extraction.** Must enforce `.detach()` on all router inputs:
     ```
     # Inside MoEClassifierHead.forward()
     router_input_eta = eta_pred.detach()
     router_input_feat = feat.detach() if use_feat_routing else None
     gate_logits = self.router_mlp(router_input_eta, router_input_feat)
     ```
   - Log per-expert load fractions every epoch: `train_moe_expert_{k}_load` for collapse monitoring.

### Must implement for Wave 5C:
4. **Curriculum training** (`--curriculum-snr`, `--curriculum-warmup-epochs`):
   - First N epochs: sample only from SNR ≥ −6 dB
   - Linearly introduce lower SNR bins over warmup period
   - By epoch N: full dataset
   - ⚠️ **PyTorch DataLoader Trap**: DataLoader workers lock in dataset state when the iterator is created. You **cannot** just change `dataset.snr_min` inside the training loop and expect the DataLoader to yield new samples. **Must rebuild the DataLoader at the start of every epoch during warmup**:
     ```
     for epoch in range(epochs):
         if args.curriculum_snr and epoch < args.curriculum_warmup_epochs:
             current_snr_min = compute_curriculum_snr(epoch, ...)
             train_dataset.set_snr_filter(min_snr=current_snr_min)
             train_loader = DataLoader(train_dataset, batch_size=..., sampler=...)
         # ... proceed with training loop ...
     ```

5. **Low-SNR oversampling** (`--snr-oversample-low`):
   - Duplicate low-SNR (< −6 dB) training samples 2x in the dataloader

### Must implement for Wave 5D:
6. **TTA evaluation** (`--tta-views`, `--tta-phase`, `--tta-shift`):
   - At inference: generate N augmented copies of each sample
   - Average logits across views before argmax
   - Augmentations: random phase rotation, circular time shift

7. **Ensemble evaluation** (`--ensemble-ckpts`):
   - Load multiple checkpoints, run inference on each
   - Average logits across models before argmax

---

## Revised Metrics and Decision Gates

Primary metrics (unchanged):
- overall test mean over all 20 SNR bins
- low band mean (`-20..−6 dB`, 8 bins)
- mid band mean (`-4..+6 dB`, 6 bins)
- high band mean (`+8..+18 dB`, 6 bins)
- class-macro (overall/low band): macro-accuracy / macro-F1

Updated promotion gates:

**Wave 5A gate** (architecture scaling):
- New oracle (W5A-7) must improve by **≥ +0.015** over previous oracle (~0.6855) **OR** exceed 0.7000 overall.
- Best blind must beat previous best (0.6408) by ≥ +0.010

**Wave 5B gate** (MoE):
- Eta-gated MoE best must beat W5A best blind by ≥ +0.005 overall with low-band gain ≥ +0.010 and high-band drop no worse than −0.003
- Oracle-gated diagnostic run:
  - if oracle-gated MoE does not beat W5A by ≥ +0.005 overall, trunk bottleneck is likely dominant
  - if oracle-gated improves but eta-gated does not, continue router/training refinements (not trunk swap yet)

**Wave 5C gate** (combined stack):
- Single-model best must reach ≥ 0.6600 overall
- ⚠️ Conservative TTA/ensemble budget: plan for +0.005 to +0.015 per technique, so 0.66 single-model → ~0.68–0.69 realistic range (not 0.70 guaranteed)

**Wave 5D gate** (final):
- TTA or ensemble result must reach ≥ 0.7000 overall

Aggressive fallback:
- If W5A oracle fails gate but W5B MoE improves low/high Pareto: continue MoE track (head decoupling can still recover ceiling gap).
- If W5A oracle fails gate **and** W5B fails to improve: scale architecture further (4-layer LSTM, wider merge conv, or Transformer encoder).
- If single-model plateau at ~0.66 after Wave 5C: explore self-supervised pretraining or larger backbone
- If TTA/ensemble gain < +0.005 per technique: investigate model diversity (different seeds alone may not be enough; try different architectures in the ensemble)

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

- Waves 1–4 addressed **training-side** levers (regularization, KD, consistency) but exhausted their returns.
- The architecture ceiling audit proves that **architecture scaling is now the primary bottleneck**.
- Wave 5A directly breaks the ceiling (bidirectional LSTM, wider classifier, new oracle).
- Wave 5B adds MoE to decouple SNR-regime decision boundaries (the original Wave 5M idea, but on a scaled architecture).
- Wave 5C combines the best of all prior techniques on the new architecture.
- Wave 5D provides a clean inference-time path (TTA + ensemble) for the final push to 0.70.
- The plan has explicit go/no-go gates at each wave, with fallback strategies if milestones are missed.

---

## Wave 5D Refresh (2026-02-25, ACTIVE): CLDNN Temporal Backbone Pivot

This section supersedes the earlier "Wave 5D inference-time TTA/ensemble" section for current execution.

### Why the pivot

- W5B.3/W5B.4 showed the oracle-routing confound was real, but MoE specialization still did not deliver a deployable gain.
- W5C-SSLv2 runs were materially worse than anchor, so the current SSL stack is not the next high-ROI lever.
- The best remaining lever is representation architecture while keeping the rest of the recipe fixed.

### Implemented code for this wave

- `--cldnn-backbone {lstm,tcn,resnet1d}` with default `lstm`
- TCN controls:
  - `--cldnn-tcn-levels`
  - `--cldnn-tcn-channels`
  - `--cldnn-tcn-kernel`
  - `--cldnn-tcn-dilation-base`
  - `--cldnn-tcn-dropout`
- ResNet1D controls:
  - `--cldnn-resnet-blocks`
  - `--cldnn-resnet-channels`
  - `--cldnn-resnet-kernel`
  - `--cldnn-resnet-dilation-cycle`
  - `--cldnn-resnet-dropout`
- `metrics.jsonl` now logs:
  - `cldnn_backbone`
  - `cldnn_tcn_levels/channels/kernel`
  - `cldnn_resnet_blocks/channels/kernel`
- `--eval-only` now persists:
  - `test_acc_by_snr.json`
  - `test_macro_summary.json`

### Active 8-run pilot matrix (no-MoE)

Common base:
- `--arch cldnn --moe-n-experts 1 --snr-mode predict`
- keep denoiser/noise/split/optimizer settings from anchor unchanged
- `--epochs 120 --early-stop-patience 15 --early-stop-min-delta 0.0005`
- seed `2016`

| ID | Host | Run name | Delta flags |
|---|---|---|---|
| T0 | Goose | `w5d_tcn_l6_c128_k3` | `--cldnn-backbone tcn --cldnn-tcn-levels 6 --cldnn-tcn-channels 128 --cldnn-tcn-kernel 3` |
| R0 | Goose | `w5d_res_b8_c128_k5` | `--cldnn-backbone resnet1d --cldnn-resnet-blocks 8 --cldnn-resnet-channels 128 --cldnn-resnet-kernel 5` |
| T1 | Athena | `w5d_tcn_l8_c128_k3` | T0 + `--cldnn-tcn-levels 8` |
| T2 | Athena | `w5d_tcn_l6_c160_k3` | T0 + `--cldnn-tcn-channels 160` |
| T3 | Athena | `w5d_tcn_l8_c160_k5` | `--cldnn-backbone tcn --cldnn-tcn-levels 8 --cldnn-tcn-channels 160 --cldnn-tcn-kernel 5` |
| R1 | Athena | `w5d_res_b10_c128_k5` | R0 + `--cldnn-resnet-blocks 10` |
| R2 | Athena | `w5d_res_b8_c160_k5` | R0 + `--cldnn-resnet-channels 160` |
| R3 | Athena | `w5d_res_b10_c160_k7` | `--cldnn-backbone resnet1d --cldnn-resnet-blocks 10 --cldnn-resnet-channels 160 --cldnn-resnet-kernel 7` |

### Decision gates (active)

1. Feasibility gate:
   - any run with `test_acc >= anchor + 0.003`
   - low-band (`-20..-6`) gain `>= +0.010`
   - high-band (`+6..+18`) drop `<= 0.003`
2. Promotion gate:
   - any run with `test_acc >= anchor + 0.008`
   - low-band gain `>= +0.015`
   - high-band drop `<= 0.003`
3. If no feasibility pass:
   - stop trunk hyperparameter sweep
   - pivot to alternate architecture family already present (`multiview`) before any new SSL wave.

---

## Wave 5D.3 (2026-02-26, ACTIVE): Ceiling-First Diagnostics (Single-Model Only)

Status from completed `w5d2` reruns:
- recipe drift is fixed (late peaks, no early collapse),
- trunk variants are near-parity but do not beat anchor yet,
- two Athena runs completed training but still need eval artifact backfill.

### Objectives

This wave is decision-focused (not broad sweeping):
1. test whether new trunks can beat historical oracle ceiling,
2. measure whether tiny blind deltas are seed noise,
3. check if a small LR retune yields immediate blind gain.

### Pre-run step (no training slots)

Backfill eval artifacts for:
- `w5d2_tcn_l6_c160_k3`
- `w5d2_tcn_l8_c128_k3`

Required files:
- `test_acc_by_snr.json`
- `test_macro_summary.json`

### Frozen base recipe (all 6 runs)

Keep the `w5d2` stable recipe fixed:
- stage schedule:
  - `--stage-a-epochs 12 --stage-b-epochs 16 --stage-a-no-cls`
  - `--stage-b1-cls2dn-scale 0.0 --stage-b2-cls2dn-scale 0.1`
- optimizer/schedule:
  - `--lr 5e-4 --min-lr 1e-5 --warmup-steps 500 --lr-decay-start-epoch 15`
- regularization/augmentation:
  - `--dropout 0.15 --label-smoothing 0.02 --aug-phase --aug-shift`
- keep denoiser/noise path unchanged,
- `--moe-n-experts 1`,
- same split/batch settings as `w5d2`.

### 6-run matrix (2 Goose + 4 Athena)

| ID | Host | Run name | Delta flags | Purpose |
|---|---|---|---|---|
| C0 | Goose | `w5d3_oracle_res_b8_c128_k5_s2016` | `--cldnn-backbone resnet1d --cldnn-resnet-blocks 8 --cldnn-resnet-channels 128 --cldnn-resnet-kernel 5 --snr-mode known --seed 2016` | Oracle ceiling test (ResNet) |
| C1 | Goose | `w5d3_oracle_tcn_l6_c160_k3_s2016` | `--cldnn-backbone tcn --cldnn-tcn-levels 6 --cldnn-tcn-channels 160 --cldnn-tcn-kernel 3 --snr-mode known --seed 2016` | Oracle ceiling test (TCN) |
| C2 | Athena | `w5d3_blind_res_b8_c128_k5_s3407` | ResNet b8 config + `--snr-mode predict --seed 3407` | Seed variance check (ResNet) |
| C3 | Athena | `w5d3_blind_tcn_l6_c160_k3_s3407` | TCN l6 c160 config + `--snr-mode predict --seed 3407` | Seed variance check (TCN) |
| C4 | Athena | `w5d3_blind_res_b8_c128_k5_lr4e4_s2016` | ResNet b8 config + `--snr-mode predict --seed 2016 --lr 4e-4` | Minimal optimization retune (ResNet) |
| C5 | Athena | `w5d3_blind_tcn_l6_c160_k3_lr4e4_s2016` | TCN l6 c160 config + `--snr-mode predict --seed 2016 --lr 4e-4` | Minimal optimization retune (TCN) |

### Hard decision gates

References:
- blind anchor: `w5a_bidir2_cls512` test `0.6409`,
- historical oracle reference: `~0.6855`.

1. Oracle headroom gate:
   - pass if `max(C0, C1) >= 0.6905` or `>= old_oracle + 0.005`,
   - secondary check: low-band (`-20..-6`) oracle gain `>= +0.010`.
2. Blind practical gate (per trunk family):
   - `mean(test_seed2016, test_seed3407) >= 0.6439`,
   - low-band gain `>= +0.010`,
   - high-band (`+6..+18`) drop `<= 0.003`.
3. Variance tie rule:
   - if `|delta vs anchor| < 0.0015` and seed std `>= 0.0015`, treat as tie (no promotion).

### Post-wave branching (locked)

1. Oracle pass + blind fail:
   - keep winning trunk family,
   - next wave = blind-gap closure only on that trunk (no new architecture sweep), start with one privileged-distillation micro-wave.
2. Oracle fail for both:
   - stop trunk-swap path,
   - revert to LSTM anchor line,
   - no further TCN/ResNet hyperparameter sweeps.
3. Oracle pass + blind pass:
   - promote winning trunk as new anchor,
   - run 3-seed confirmation on promoted candidate.

---

## Wave 5E.1 / 5E.2 (2026-02-26, ACTIVE): Deconfounded Waveform Diffusion Front-End

### Why this wave

- Prior MoE/SSL/trunk-swap waves did not produce reliable blind gains.
- We now isolate feasibility of a **waveform diffusion denoiser front-end** with a frozen backend before any full-stack coupling.

### Implemented API additions

Training/eval flags now available in `train.py`:

- core:
  - `--dn-diff-enable`
  - `--dn-diff-target {v,eps}`
  - `--dn-diff-train-timesteps`
  - `--dn-diff-beta-start`
  - `--dn-diff-beta-end`
- split t-source:
  - `--dn-diff-train-t-start-source {snr_pred,fixed}`
  - `--dn-diff-eval-t-start-source {snr_pred,snr_true,fixed}`
  - `--dn-diff-fixed-t-start`
  - `--dn-diff-snr2t-scale`
  - `--dn-diff-snr2t-bias`
- eval sampler:
  - `--dn-diff-eval-mode {ddim,onestep}`
  - `--dn-diff-eval-steps`
  - `--dn-diff-ddim-eta`
  - `--dn-diff-multisample`
- protection/masking:
  - `--dn-diff-low-snr-thresh`
  - `--dn-diff-high-snr-margin`
  - `--dn-diff-hard-bypass-high-snr`
  - `--dn-diff-apply-lowband-only-train`
  - `--dn-diff-loss-snr-lo`
  - `--dn-diff-loss-snr-hi`
- loss/control:
  - `--dn-diff-freeze-classifier`
  - `--init-ckpt`
  - `--init-ckpt-source {auto,model,ema}`
  - `--lambda-dn-diff`
  - `--lambda-dn-recon`
  - `--lambda-dn-feat-align`
  - `--lambda-dn-logit-align`
  - `--dn-diff-align-teacher {none,frozen,ema}`
  - `--dn-diff-feat-align-start-epoch`
  - `--dn-diff-logit-align-start-epoch`
- diagnostics:
  - `--dn-diff-cond-diagnostic {none,zero,shuffle}`
  - `--dn-diff-force-deterministic-multisample`

`metrics.jsonl` now includes additive diffusion fields:
- config: `dn_diff_enabled`, `dn_diff_target`, `dn_diff_train_t_source`, `dn_diff_eval_t_source`, `dn_diff_eval_mode`, `dn_diff_eval_steps`, `dn_diff_ddim_eta`, `dn_diff_multisample`, etc.
- warm-start trace: `init_ckpt`, `init_ckpt_source_used`
- train losses: `train_loss_dn_diff`, `train_loss_dn_recon`, `train_loss_dn_feat_align`, `train_loss_dn_logit_align`
- diagnostics: `dn_diff_t_start_mean`, `dn_diff_t_start_std`, `dn_diff_active_frac_low/mid/high`
- validation mirrors: `val_dn_diff_*`
- per-epoch SNR trajectory file: `val_acc_by_snr_history.jsonl`

### E.1 execution corrections (2026-02-26)

- Full incident log: `W5E1_INCIDENT_REPORT_2026_02_26.md`.
- Root causes observed in first E.1 attempt:
  - frozen-classifier runs launched without reliable warm-start;
  - eval path using EMA shadow before stable EMA state in warm-started runs.
- Code corrections now in place:
  - warm-start source selector: `--init-ckpt-source {auto,model,ema}`;
  - in `auto`, frozen-classifier runs prefer checkpoint EMA state when available;
  - EMA eval is only used after `global_step >= ema_start`.
- Operational protocol for current E.1 reruns:
  - always pass `--init-ckpt` for frozen-classifier runs;
  - set `--ema-decay 0` for deconfounded feasibility runs;
  - use schedule coverage settings that avoid low-band timestep saturation
    (`--dn-diff-train-timesteps 500` or `--dn-diff-beta-end 0.10`);
  - run 1-2 epoch smoke before launching full E.1 matrix.

### E.1 schedule-coverage precheck (required before full matrix)

Run two short smokes and keep all non-schedule flags fixed:
- Variant A: `--dn-diff-train-timesteps 500 --dn-diff-beta-end 0.02`
- Variant B: `--dn-diff-train-timesteps 100 --dn-diff-beta-end 0.10`

Select the variant with better low-band (`-14..-6`) without high-band damage, then use that single schedule setting for full E.1.

### Wave 5E.1 run matrix (2 Goose + 4 Athena)

Base:
- current stable single-model CLDNN recipe,
- `--arch cldnn --moe-n-experts 1`,
- `--epochs 80 --early-stop-patience 12 --early-stop-min-delta 0.0005`.

| ID | Host | Run name | Purpose |
|---|---|---|---|
| E0 | Goose | `w5e1_ctrl_frozen_residual_s2016` | matched frozen control (no diffusion) |
| E1 | Goose | `w5e1_diff_core_pred_evalpred_s2016` | blind diffusion feasibility |
| E2 | Athena | `w5e1_diff_core_pred_evaloracle_s2016` | oracle-eval-only t-start diagnostic |
| E3 | Athena | `w5e1_diff_core_pred_evalfixed30_s2016` | fixed-t sanity |
| E4 | Athena | `w5e1_diff_core_pred_evalpred_recon003_s2016` | recon ablation |
| E5 | Athena | `w5e1_diff_core_pred_evalpred_featalign_s2016` | delayed feature-align probe |

### Mandatory non-run diagnostics

1. Conditioning usefulness (E1 checkpoint):
- `--dn-diff-cond-diagnostic none|zero|shuffle`

2. One-step vs DDIM mismatch (E1/E2 checkpoints):
- `--dn-diff-eval-mode onestep`
- `--dn-diff-eval-mode ddim --dn-diff-eval-steps 8`

### Hard gates

- Feasibility pass (any E1..E5):
  - low (`-14..-6`) `>= E0 + 0.010`
  - high (`+6..+18`) drop `<= 0.003`
  - overall `>= E0 - 0.002`
- Mapping gate:
  - `E2 - E1 >= +0.005` overall or `+0.010` low ⇒ SNR→t mapping bottleneck.

### Wave 5E.2 trigger

Only if E.1 passes:
- keep winning E.1 core config,
- add feature align, then logit align, then late partial unfreeze,
- run multisample sweeps (`N=3`, then `N=5`) with stochastic DDIM only.

### Wave 5E.2 update (Option D primary, 2026-02-26)

Decision:
- use **hybrid denoiser objective** as default (diffusion regularizer + task-aware CE),
- treat pure frozen E.1 as diagnostic-only (not final optimization target).

Rationale:
- pure diffusion objective (`lambda_dn_diff` only) improved denoiser loss but plateaued below matched control,
- adding task-aware signal (`lambda_dn_cls`) is required to optimize for classification utility,
- keeping diffusion loss nonzero reduces degenerate classifier-hack behavior.

New Option-D schedule controls in `train.py`:
- `--dn-diff-cls-warmup`
- `--dn-diff-cls-ramp`
- `--dn-diff-diff-warmup`
- `--dn-diff-diff-ramp`
- `--dn-diff-diff-final-scale`

Per-epoch effective loss weights now logged:
- `lambda_dn_cls_eff`
- `lambda_dn_diff_eff`

Recommended default schedule for first E.2 wave:
- `--lambda-dn-cls 0.20`
- `--dn-diff-cls-warmup 12`
- `--dn-diff-cls-ramp 16`
- `--lambda-dn-diff 1.0`
- `--dn-diff-diff-warmup 12`
- `--dn-diff-diff-ramp 16`
- `--dn-diff-diff-final-scale 0.20`

Interpretation:
- Stage A: classifier signal off, full diffusion regularization.
- Stage B: classifier signal ramps in while diffusion weight decays to regularizer level.
- Post-Stage B: task-aware objective dominates, diffusion remains as stability prior.
