# Architecture Plan V3.1: Noise-Fraction-Aware Denoising CLDNN (Blind Deployment)

## 1) Why We Are Pivoting

This plan reflects current evidence from our runs:

- `cldnn_base_v2_tuned` remains the strongest reliable baseline.
- `cldnn_film_predict` showed gains, especially at low/mid SNR, so conditioning is useful.
- SNR-path consistency has not produced stable net gains after bug fixes.
- Standard SupCon (`supcon_l01`, `supcon_l02`) is stable but remains below baseline.

Conclusion: we should keep the strong CLDNN core and pivot to a method that directly improves corrupted signal structure, while preserving blind inference.

---

## 2) Non-Negotiable Design Constraints

1. **Blind inference only**: no oracle SNR/noise labels at deployment.
2. **Physics-consistent training**: noise operations must match AWGN math.
3. **Ablation-first development**: one major change at a time.
4. **Keep CLDNN backbone**: extend, do not replace.
5. **Complexity budget**: every added module must justify itself with measurable gain.

---

## 3) Signal/Math Foundation (RMS-Norm Trap Fixed)

Assume received baseband sample:

- `x[n] = s[n] + w[n]`
- `E[s w*] = 0`
- `SNR_lin = P_signal / P_noise`
- `P_total = P_signal + P_noise`

From known training SNR (`s_db`) and measured total power (`P_total`), derive:

- `SNR_lin = 10^(s_db / 10)`
- `P_signal = P_total / (1 + 1 / SNR_lin)`
- `P_noise = P_total / (1 + SNR_lin)`

Critical identifiability note:

- With per-sample `--normalize rms`, `P_total` is close to constant.
- In that regime, regressing absolute `P_noise` can collapse into a weak SNR-proxy problem.

Therefore the primary conditioning target is **noise fraction**, not raw noise power:

- `rho = P_noise / P_total = 1 / (1 + SNR_lin)` in `(0,1)`
- `eta = logit(rho) = log((rho + eps) / (1 - rho + eps))`

We supervise `eta` (or `rho`) and use it for FiLM/denoiser conditioning.  
Optional raw-power path is still useful, but not required for identifiability.

---

## 4) Final Architecture Decision

## 4.1 High-Level Block Diagram

Training uses two coupled paths:

1) **Paired denoise supervision path**

`x` -> `degrade(x, s_db, Delta)` -> `x_low`

`x_low` -> `[AnalyticNoiseProxy + NoiseFractionNet]` -> `eta_hat_low`

`x_low` -> `[DenoiserUNet1D conditioned on eta_hat_low]` -> `x_dn_low`

`(x_low, x_dn_low, x)` -> `L_np + L_dn`

2) **Deployment-matched classifier path (default)**

`x` -> `[AnalyticNoiseProxy + NoiseFractionNet]` -> `eta_hat`

`x` -> `[DenoiserUNet1D conditioned on eta_hat]` -> `x_dn`

`[x, x_dn]` -> `[CLDNN + Expert branch + FiLM(eta)]` -> `logits`

Optional classifier robustness augmentation:

- With low probability (`p_cls_aug <= 0.3`), classifier can consume `[x_low, x_dn_low]` instead of `[x, x_dn]`.

Inference path (blind):

`x_in` -> `[AnalyticNoiseProxy + NoiseFractionNet]` -> `eta_hat_in`

`x_in` -> `[DenoiserUNet1D conditioned on eta_hat_in]` -> `x_dn_in`

`[x_in, x_dn_in]` -> classifier

Conditioning-alignment rule (must hold):

- The signal fed to the denoiser and the signal fed to the noise head must match (`x_low` with `x_low`, `x_in` with `x_in`).
- No cross-signal conditioning (prevents train/inference leakage).

Outputs during training:

- `logits`
- `eta_hat` / `eta_hat_low`
- `x_dn` / `x_dn_low`

## 4.2 Components

1. **NoiseFractionNet (new, hybrid)**
   - Primary prediction target: `eta_hat = logit(rho_hat)`.
   - Input includes:
     - learned branch over IQ, and
     - analytic proxy `rho0` (scale-invariant; from local difference energy).
   - Define analytic proxy explicitly:
     - `e = mean(|x[n] - x[n-1]|^2) / (mean(|x[n]|^2) + eps)`.
     - `rho0 = g_cal(e)`, where `g_cal` is a monotonic calibration fit on train data
       (isotonic regression or a small logistic curve), then clamp to `[1e-4, 1-1e-4]`.
   - Hybrid form: `eta_hat = clamp(logit(rho0) + delta_eta_nn, eta_min, eta_max)`.
   - Supervised with `eta_target` from SNR labels during training.
   - Deployment uses prediction only (no labels).

2. **DenoiserUNet1D (new, lightweight)**
   - 1D U-Net over IQ channels (`2 x L`), residual form:
     - `x_denoised = x_input + delta(x_input, cond)`
   - Conditioned on `eta_hat` embedding.
   - Lightweight by design (to avoid overfitting and latency explosion).
   - Optional soft high-SNR residual suppression:
     - `x_dn = x + (1 - m_hi_eta(eta_hat)) * delta(x, eta_hat)`
     - `m_hi_eta(eta_hat) = sigmoid((eta_hi - eta_hat) / tau_eta)`,
       with `eta_hi = -ln(10)` (about `+10 dB`) and `tau_eta = 0.46` (about `2 dB` transition).

3. **Classifier (existing CLDNN, dual-path mandatory)**
   - Keep current CLDNN backbone and attention pooling.
   - Keep optional expert branch (cyclostationary stats) because it is low-SNR friendly.
   - Input both raw and denoised views (`[x_raw, x_dn]`) via **channel concat** for initial implementation and ablations.
   - Consider late fusion only if channel-concat saturates.
   - Replace SNR-FiLM conditioning scalar with `eta_hat`.

4. **FiLM on noise fraction (modified)**
   - Use **`eta_hat` only** as conditioning input (single canonical variable).
   - Do not feed `rho_hat` or raw power directly into FiLM/denoiser conditioning paths.
   - Keep identity initialization (`gamma ~ 1`, `beta ~ 0`) for stability.
   - Soft conditioning only; no hard thresholds.

---

## 5) Training Objective (Stage-Wise)

## 5.1 Core Losses

1. **Classification loss**
   - `L_cls = CE(logits, y)` (optionally focal only if needed).

2. **Noise-fraction regression loss**
   - `rho_target = 1 / (1 + 10^(s_db / 10))`.
   - `rho_target = clamp(rho_target, 1e-4, 1 - 1e-4)`.
   - `eta_target = logit(rho_target)`, then `eta_target = clamp(eta_target, -8.0, +5.5)`.
   - Predicted `rho_hat`/`eta_hat` are clamped to the same valid domain before loss.
   - `L_np = SmoothL1(eta_hat, eta_target)`.
   - Primary supervision is on the same signal used for denoising (`x_low` pathway).

3. **Denoising loss (paired by controlled degradation)**
   - Create `x_low` from current sample via calibrated AWGN degradation (incremental-noise math):
     - **Post-R1/R2 default (goal-band targeted):** sample degraded SNR from the overlap of
       `[s_db - 8, s_db - 2]` with the target band `[-14, -6]`:
       - `s_lo = max(s_db - 8, -14)`
       - `s_hi = min(s_db - 2, -6)`
       - if `s_lo < s_hi`: sample `s_new_db ~ Uniform(s_lo, s_hi)`
       - else (no overlap): fallback to legacy rule `s_new_db = clamp(s_db - U(2,8), min=snr_floor)` (or skip paired loss for that sample)
       - `Delta_actual = s_db - s_new_db`
     - **Legacy/compatibility rule:** `Delta ~ Uniform(2, 8) dB`, then `s_new_db = clamp(s_db - Delta, min=snr_floor)`.
     - **Linear-scale rule (mandatory):** convert dB to linear **before** any subtraction:
       - `SNR_orig_lin = 10^(s_db/10)`, `SNR_new_lin = 10^(s_new_db/10)`.
       - Never subtract dB values to get power deltas.
     - `P_total = mean(|x|^2)`.
     - `P_signal = P_total / (1 + 1 / SNR_orig_lin)`.
     - `P_noise_orig = P_signal / SNR_orig_lin`.
     - `P_noise_new = P_signal / SNR_new_lin`.
     - `P_noise_add = clamp(P_noise_new - P_noise_orig, min=0)`.
     - `x_low = x + sqrt(P_noise_add) * eps`, `eps ~ N(0,1)`.
   - Do not treat `P_total` as `P_signal`; that silently distorts the intended `Delta`.
   - Apply paired denoise/noise-head losses only when `Delta_actual >= 2 dB`.
   - Train denoiser to recover the less-noisy target:
     - `L_dn = ||D(x_low) - x||_1` (or Charbonnier)

4. **Conditional identity regularization (high SNR only)**
   - Prevent hallucination only where denoising should be near-identity:
     - `L_id = m_hi(s_db) * ||D(x) - x||_1`
   - Use a **soft high-SNR mask**:
     - `m_hi(s_db) = sigmoid((s_db - 10) / 2)`.

5. **High-SNR teacher distillation (AMC preservation)**
   - Freeze best baseline classifier as teacher.
   - For high-SNR samples, preserve class evidence after denoising:
     - `L_kd = m_hi * KL(softmax(z_teacher(x)/T) || softmax(z_student([x, D(x)])/T))`
   - Initial temperature: `T = 2`.

6. **Feature preservation loss (early backbone)**
   - Preserve discriminative local structure:
     - Default paired form:
       - `L_feat = ||f_early([x_low, D(x_low)]) - f_early([x_low, x])||_1`
     - Optional deployment-form high-SNR regularizer:
       - `L_feat_hi = m_hi * ||f_early([x, D(x)]) - f_early([x, x])||_1`
   - `f_early` from a frozen early CLDNN block.

Total (baseline joint objective):

- `L = L_cls + lambda_np * L_np + lambda_dn * L_dn + lambda_id * L_id + lambda_kd * L_kd + lambda_feat * L_feat`

## 5.2 What We Do With SupCon/Consistency

- **Standard SupCon**: disabled by default (current evidence: no gain).
- **SNR-path consistency**: disabled by default in V3.
- Optional future add-on only after V3 is stable:
  - **Cross-SNR SupCon** (same class, different SNR positives), very small weight.

---

## 6) Staged Training Protocol

## Stage A: Noise Head + Denoiser Bootstrap (short)

Goal: learn a stable denoiser + noise estimator before classifier coupling.

- Train NoiseFractionNet + Denoiser with `L_np + L_dn + L_id`.
- Classifier frozen or omitted.
- Force **hard-SNR coverage** in this stage (not only easy samples), so denoiser does not learn identity-only behavior.
- 5-10 epochs.

## Stage B: Classifier Warm Start With Frozen/Slow Denoiser

Goal: prevent classifier collapse while denoiser is still adapting.

- Unfreeze classifier.
- Keep denoiser LR lower (or partially frozen encoder).
- Use explicit two-step gradient policy:
  - Stage B1 (first 50%): block `L_cls` gradients into denoiser (`x_dn` detached on classifier path).
  - Stage B2 (second 50%): enable `L_cls` gradients with multiplier `g_cls2dn = 0.1`.
- Optimize `L_cls + lambda_np * L_np + small lambda_dn * L_dn + lambda_id * L_id + lambda_kd * L_kd + lambda_feat * L_feat`.
- 10-20 epochs.

## Stage C: Joint Fine-Tuning

Goal: maximize end performance.

- Train all modules jointly.
- Full loss.
- Use early stop on validation accuracy and low-SNR aggregate metric.

---

## 7) Recommended Initial Hyperparameters

Global defaults:

- `rho` clamp: `[1e-4, 1 - 1e-4]`.
- `eta` clamp: `[-8.0, +5.5]`.
- Conditioning variable (FiLM + denoiser): `eta_hat` only.
- Paired degradation (legacy/general): `Delta ~ Uniform(2, 8) dB`.
- Paired degradation (post-R1/R2 `L_feat` ablations): target `s_new_db` in overlap of `[s_db-8, s_db-2]` and `[-14, -6]`, with fallback when no overlap.
- Apply paired denoise/noise-head losses only if `Delta_actual >= 2 dB`.
- High-SNR soft mask: `m_hi(s_db) = sigmoid((s_db - 10) / 2)`.
- Distillation temperature: `T = 2`.
- Optional denoiser high-SNR blend constants: `eta_hi = -ln(10)`, `tau_eta = 0.46`.
- Classifier degraded-input augmentation probability cap: `p_cls_aug <= 0.3`.
- Denoiser base channels: `32` (increase to `48/64` only if clear gain).

Stage-wise loss defaults:

- **Stage A (bootstrap)**
  - `lambda_dn = 1.0`
  - `lambda_np = 0.1`
  - `lambda_id = 0.02`
  - `lambda_kd = 0.0`
  - `lambda_feat = 0.0`

- **Stage B (warm start classifier)**
  - `lambda_dn = 0.3`
  - `lambda_np = 0.1`
  - `lambda_id = 0.03`
  - `lambda_kd = 0.05`
  - `lambda_feat = 0.02`

- **Stage C (joint fine-tune)**
  - `lambda_dn = 0.1`
  - `lambda_np = 0.05`
  - `lambda_id = 0.02`
  - `lambda_kd = 0.1`
  - `lambda_feat = 0.03`

- **`L_feat`-isolation override for R3a/R3b/R3c**
  - Set `lambda_kd = 0.0` to isolate `L_feat` impact.
  - Re-enable KD only in follow-up run `R3d`.

Training setup:

- Keep baseline-proven setup for fair comparison:
  - `batch_size=512`, `lr=5e-4`, `min_lr=1e-5`, `warmup_steps=500`, `lr_decay_start_epoch=15`.
- Learning-rate policy for next runs:
  - Keep `lr=5e-4` fixed while testing architecture/regularization changes (isolates effects cleanly).
  - Run LR sweep (`4e-4`, `5e-4`) only after regularization settings are stabilized.
- Augmentation policy for next runs:
  - Keep `--aug-phase` enabled (non-negotiable default).
  - Keep `--aug-shift` enabled for denoiser runs to preserve time-shift invariance.
- Regularization defaults for denoiser ablations moving forward:
  - `dropout=0.20`, `label_smoothing=0.02`, `mixup_alpha=0.2` (with default `mixup_prob`).
- After confirming a gain, run a separate throughput sweep for larger batch.

---

## 8) Ablation Matrix (Strict Order)

All runs compared against the same baseline recipe and same split.

1. **A0**: Baseline CLDNN (reference)
2. **A1**: Baseline + noise-fraction FiLM only (no denoiser)
3. **A2**: Baseline + denoiser only, single-path input (`x_dn`)
4. **A2b**: Same training as A2, but denoiser bypassed at inference (sanity check for regularization-only effects)
5. **A3**: Baseline + denoiser + mandatory dual-path input (`[x_raw, x_dn]`)
6. **A4**: A3 + noise-fraction FiLM
7. **A5**: A4 + expert feature branch
8. **A6 (optional)**: A5 + cross-SNR SupCon (tiny weight)

Stop criteria:

- Keep a module only if it improves:
  - overall accuracy (non-negative vs baseline), and/or
  - low-SNR band (`-14` to `-6 dB`) by at least `+1.0 pp`,
  - while high-SNR (`>= +10 dB`) drop is no worse than `-0.3 pp`.

Execution status snapshot (Feb 2026):

- A0 complete (reference).
- A1 complete (`eta` known + `eta` predicted variants; both below A0).
- A2 complete (single-path denoiser; small net gain, strongest at low SNR).
- A2b complete (bypass-eval sanity check; clearly below A2).
- A3 complete (dual-path denoiser; current best accuracy).
- A4/A5 pending as the official next matrix steps.

---

## 9) Implementation Plan by File

## `model.py`

- Add `AnalyticNoiseProxy` (`rho0` from local difference energy).
- Add lightweight `NoiseFractionNet` (residual correction on `logit(rho0)`).
- Add `DenoiserUNet1D` (residual IQ denoiser).
- Extend CLDNN wrapper to optionally:
  - run denoiser preprocessor,
  - accept dual-path input (`x_raw` + `x_dn`),
  - consume `eta_hat` conditioning for FiLM,
  - expose auxiliary outputs for losses.

## `train.py`

- Add CLI flags:
  - denoiser enable, channels, loss weights,
  - noise-fraction head enable, weight,
  - stage schedule toggles.
- Add target builder for `eta_target = logit(rho_target)` from `snr_db`.
- Implement `degrade(x, s_db, Delta)` with incremental-noise math (`P_noise_add`), not `P_total`-as-signal.
- Add calibration fit for analytic proxy map `g_cal: e -> rho0` (monotonic).
- Enforce conditioning-alignment rule in code path (`noise_head` input must match denoiser input).
- Add staged optimizer logic (or param-group LR multipliers).
- Add explicit Stage-B classifier->denoiser gradient schedule (`detach` then `g_cls2dn`).
- Log noise-head calibration diagnostics every eval epoch.
- Keep old consistency/SupCon flags, but default them off for V3 runs.

## `data.py`

- Add optional support to preserve/access raw-power information for noise estimation path.
- Keep current behavior as default to avoid breaking existing runs.

---

## 10) Complexity Reduction Policy

Move non-performing ideas to "inactive defaults" (not deleted yet):

- SNR-path consistency (inactive by default)
- Standard SupCon (inactive by default)
- STFT multi-view fusion (inactive by default)

Rationale: preserve reproducibility while reducing active complexity during iteration.

---

## 11) Risks and Mitigations

1. **Denoiser over-smooths discriminative details**
   - Mitigation: residual design + high-SNR masked identity + teacher KD + feature preservation + dual-path input.

2. **Noise estimator becomes a weak SNR proxy**
   - Mitigation: predict `eta=logit(rho)` instead of raw `P_noise`; hybrid analytic+learned estimator; evaluate calibration.

3. **Conditioning leakage (train/inference mismatch)**
   - Mitigation: strict rule that denoiser-conditioning head sees the same signal as denoiser input.

4. **Joint training instability**
   - Mitigation: staged schedule, lower denoiser LR in Stage B, delayed gradient flow from classifier loss.

5. **No net gain over baseline**
   - Mitigation: strict ablation gates, include A2b sanity ablation, keep modules rollback-safe.

---

## 12) Success Criteria

A V3 configuration is considered successful if it achieves:

- better or equal overall test accuracy vs baseline, and
- low-SNR gain (`-14` to `-6 dB`) of at least `+1.0 pp`, and
- high-SNR (`>= +10 dB`) regression no worse than `-0.3 pp`.

If not met, we keep CLDNN baseline and retain only proven components.

---

## 13) Monitoring and Diagnostics (First-Class)

Track these every validation epoch:

1. **Noise-head correlation**
   - Pearson and Spearman correlation between `eta_hat` and `eta_target`.

2. **Calibration by SNR bucket**
   - For each SNR bin: mean/std of `eta_hat`, mean `eta_target`, and absolute gap.
   - Reliability-style plot over `eta` bins.

3. **Monotonicity**
   - `eta_hat` should monotonically decrease with SNR (strong negative rank correlation).

4. **Ablation interpretation guardrail**
   - Compare A2 vs A2b to separate true inference-time denoiser gain from auxiliary regularization-only gain.

If calibration quality is poor, pause downstream ablations and fix noise-head calibration first.

---

## 14) Immediate Next Action (Execution Update)

Current evidence after A0-A3:

- Complexity increased substantially with denoiser path, while net overall gain is modest.
- Gains are concentrated in low SNR; mid-SNR regression appears when regularization is weak.
- Therefore, keep official ablation order, but apply low-risk regularization fixes before deeper architectural changes.

Next-run priorities:

1. **LR decision**
   - Keep `lr=5e-4` unchanged in immediate next runs for fair attribution.
   - Do not change LR and multiple knobs simultaneously.

2. **Augmentation decision**
   - Keep `--aug-phase` ON.
   - Keep `--aug-shift` ON (restore baseline-consistent invariance).

3. **Two practical runs to queue next (parallel screens)**
   - Run R1: A4 recipe + `dropout=0.20` + `mixup_alpha=0.2` + `lambda_id=0.05` + both phase/shift aug.
   - Run R2: same as R1 + expert branch + SNR-balanced sampling (`--snr-balanced --snr-balance-power 0.5`).

4. **Decision gate**
   - If R1 >= current A3 and improves mid-SNR dip, keep tuned regularization as new default.
   - If R2 provides additional low-SNR gain without high-SNR regression, promote it as A5 default.

5. **Only then run LR sweep**
   - Compare `lr=4e-4` vs `5e-4` on the winning recipe; keep whichever improves best-val and low-SNR average together.

6. **If V3.2 plateaus after A5: add targeted "outside-the-box" regularizers**
   - Prerequisite/status note: A2b has already been run and is below A2, so denoiser has demonstrated inference-time value (not only train-time regularization).
   - Add-ons below are optional and should be introduced one at a time.

   - **Hierarchical auxiliary heads (first choice, lower risk):**
     - Add one coarse taxonomy head on shared features (e.g., analog vs digital, or small modulation families).
     - Train jointly with a small weight (`lambda_aux` around `0.05-0.15`) during Stage B/C.
     - Goal: improve low-SNR robustness by enforcing coarse-grained separability that survives heavy noise.

   - **Confusion-pair reweighting in low-SNR band (second choice, higher risk):**
     - Compute confusion statistics only for `-14..-6 dB` on the train split (or train-EMA), then upweight hardest class pairs.
     - Apply reweighting only in that SNR band; cap pair weights (e.g., max `2x`) and warm up before enabling.
     - Do not use validation confusion to set train weights (avoid leakage).
     - Goal: target persistent low-SNR error pairs more directly than global focal scaling.

---

## 15) Post-R1/R2 Execution Plan: Implement `L_feat` Properly

Status note (current): R1 and R2 are in progress. This section is the locked implementation plan to execute immediately after those runs finish.

### 15.1 Why this is the next highest-value step

- Current denoiser supervision in code is `L_dn` (waveform L1) + `L_id` (high-SNR identity mask).
- `L_feat` is specified in this plan, but not implemented yet.
- A2 > A2b confirms denoiser has true inference-time contribution, so improving denoiser/classifier alignment is likely higher value than adding another large module first.
- Primary hypothesis: if denoiser is smoothing discriminative cues, feature-preservation can recover low-SNR class evidence without forcing architecture expansion.

### 15.2 Exact loss form to implement first (paired form only)

Default paired form (first implementation):

- `L_feat = ||f_early([x_low, D(x_low)]) - f_early([x_low, x])||_1`

Implementation notes:

- For dual-path runs, use the exact paired concatenation above.
- For single-path runs, use equivalent pair:
  - `L_feat_single = ||f_early(D(x_low)) - f_early(x)||_1`
- Keep optional high-SNR deployment regularizer (`L_feat_hi`) deferred until paired form is validated.

### 15.3 `model.py` scope (minimal, ablation-safe)

Add a frozen early-feature encoder utility in `CLDNNAMC`:

- `build_feat_encoder()`:
  - Snapshot early CNN blocks used before temporal reduction (`conv_iq`, `conv_i`, `conv_q`, `conv_fuse`, `conv_merge`).
  - Freeze snapshot params (`requires_grad=False`) and force `eval()` mode.
- `has_feat_encoder` property:
  - Fast guard to ensure the feature encoder exists before computing `L_feat`.
- `early_features(x)`:
  - Accept classifier-form input (`2ch` single-path or `4ch` dual-path).
  - Return early feature map after `conv_merge` (before temporal reduction/LSTM).
  - Run in float32 for stability under AMP; no dropout usage in this path.

Rationale:

- Use a dedicated frozen snapshot (not shared live layers) so training cannot "move the ruler."
- Preserve gradient flow to denoiser input while keeping feature-extractor weights fixed.

### 15.4 `train.py` scope (where to wire loss)

1. Add CLI knobs:
   - `--lambda-feat` (default `0.0`)
   - `--feat-ramp-epochs` (default `5`)
   - optional `--feat-encoder-ckpt` (default `None`)

2. Add helper:
   - `get_lambda_feat(epoch, args)`:
     - `0` during Stage A
     - linear ramp over `feat_ramp_epochs` after Stage A
     - cap at `lambda_feat`

3. Build/snapshot `f_early`:
   - At Stage-B start (or epoch 0 if Stage A disabled), call `model.build_feat_encoder()`.
  - For `L_feat` ablations, lock one fixed source checkpoint via `--feat-encoder-ckpt` (recommended: the selected A0 baseline checkpoint for this experiment family).
  - Do not use "current model state" as the default perceptual ruler in R3 ablations; keep the ruler fixed across runs.

4. Compute `loss_feat` in both CLDNN denoiser branches (AMP and non-AMP code paths):
   - Reuse existing paired tensors (`x_low_dn`, `x_dn_low`, `x_dn_ref`).
   - Dual-path:
     - pred input: `[x_low_dn, x_dn_low]`
     - target input: `[x_low_dn, x_dn_ref]`
   - Single-path:
     - pred input: `x_dn_low`
     - target input: `x_dn_ref`
   - `feat_tgt = f_early(...).detach()`
   - `loss_feat = mean(abs(feat_pred - feat_tgt))`
   - Add to total loss with ramped coefficient.

5. Logging:
   - Add `lambda_feat` and `loss_feat` to epoch metrics/logging.
   - Add a small counter/summary to confirm non-zero activation epochs.

### 15.5 Hardening checklist (must-pass)

- `f_early` frozen: all params `requires_grad=False`
- `f_early` in `eval()` mode
- target features detached
- `loss_feat` uses mean reduction (scale-safe)
- `lambda_feat` starts small and ramps
- compute feature path in float32 under AMP
- keep grad clipping enabled
- verify `loss_feat` non-zero only when intended

### 15.6 Post-R1/R2 run matrix (strict order)

Keep winner recipe fixed (no LR/augmentation changes while testing `L_feat`).

1. **R3a (primary)**: winner + `L_feat` paired form
   - `lambda_feat=0.02`, `feat_ramp_epochs=5`, `lambda_kd=0.0`
   - fixed `feat_encoder_ckpt` (same checkpoint for all R3* runs)
2. **R3b (if R3a is non-negative overall)**: same recipe, stronger feature weight
   - `lambda_feat=0.03`, same ramp, `lambda_kd=0.0`
3. **R3c (only if needed)**: same as best R3x + degradation-policy ablation
   - compare target-band degradation vs legacy `Delta ~ U(2,8)` while keeping all else fixed
4. **R3d (follow-up, only after best R3x is chosen)**: re-enable KD
   - turn on KD and measure incremental gain on top of the best `L_feat`-only recipe

Do not combine with additional new modules in this phase.

### 15.7 Decision gates for promotion/rejection

Promote `L_feat` as default only if all hold:

- overall accuracy is non-negative vs the matched no-`L_feat` recipe
- low-SNR (`-14..-6 dB`) improves by at least `+1.0 pp`
- high-SNR (`>= +10 dB`) regression no worse than `-0.3 pp`

Reject or rework if:

- `loss_feat` destabilizes training (spikes/NaNs), or
- low-SNR gain is negligible and confidence diagnostics show no structure-preservation benefit.

### 15.8 Explicit non-goals for this phase

- Do **not** switch denoiser `L_dn` from L1 to L2 in this ablation.
- Do **not** add external teacher-KD in `R3a/R3b/R3c` (isolate effect of `L_feat` first).
- Do **not** run LR sweep until `L_feat` on/off conclusion is established.

