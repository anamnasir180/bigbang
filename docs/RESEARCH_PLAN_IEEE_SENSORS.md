# Re-planning "Unconscious vs. Pushed" Fall-Cause Classification for IEEE Sensors

**Constraint:** we cannot collect new data (repeating push/collapse falls is unsafe).
**Goal:** reuse the existing 200 trials, and optionally public datasets, to produce a study
that holds up to IEEE Sensors review, and re-run all experiments under a correct protocol.

---

## 1. Why the current manuscript would be rejected (reviewer's-eye audit)

| # | Problem in current draft | Why reviewers reject it | Can it be fixed without new data? |
|---|---|---|---|
| R1 | Only **2 participants** (24 y / 50 kg, 29 y / 55 kg) | No evidence the model generalizes to a new person; the model may learn *who* fell rather than *why* | **Partially.** Report leave-one-subject-out (LOSO) results honestly, add external public data (Sec. 4), and frame the work as a pilot/feasibility study |
| R2 | One random train/val/test split (≈129/32/39) | Trials from the same person and session end up on both sides of the split, which inflates accuracy. With 39 test samples, 92% ± ~9% is not a stable number | **Yes.** Nested repeated CV, LOSO, bootstrap confidence intervals, permutation test |
| R3 | The text implies hyperparameters were chosen with the validation/test sets in view (e.g., k-NN k=1) | Optimistic bias | **Yes.** Nested CV: tune only inside the inner loop |
| R4 | Fall windows cut out by **visual inspection** | Not reproducible; a human who knows the label is choosing the window (label leakage) | **Yes.** Automatic impact-anchored segmentation (Sec. 3.2) |
| R5 | Features are only the 121 raw magnitude samples | Axis/orientation information is discarded; there is no signal-processing or sensor contribution; results can't be interpreted | **Yes.** Recompute from raw x/y/z (and gyro, if logged; see 2.1) using phase-based, physically meaningful features |
| R6 | No sensor-level contribution | IEEE Sensors J. desk-rejects "ML applied to a public/phone dataset" papers that say nothing about the sensor itself | **Yes.** Sensor-configuration ablations: sampling rate, range/saturation, resolution, noise, axes, placement (Sec. 3.5) |
| R7 | Literature comparison table lists fall-**detection** accuracies (99%) next to our fall-**cause** accuracy | Compares different tasks; reviewers see this as misleading | **Yes.** Remove it. Compare against baselines we run ourselves on the same data |
| R8 | "Syncope" simulated as *holding the head or reaching for support* | Real syncope involves **loss of muscle tone**. Reviewers will question whether the class is valid | **Reframe:** call the class "prodrome-then-collapse" (symptomatic self-initiated collapse) vs. "externally perturbed". State the limitation explicitly |
| R9 | Forensic "John murder" scenario; legal/assault paragraphs | Unscientific; claims a use case the data cannot support | **Yes.** Remove it. Motivate with emergency triage (syncope needs medical work-up; a push/trip does not) |
| R10 | No ethics/IRB statement even though participants were pushed | IEEE requires one; can lead to desk rejection | **Must be checked.** Obtain the approval letter or a retrospective statement from the UoG ethics committee. Do not submit without it |
| R11 | Methods section appears twice; duplicated figures; heatmaps for every hyperparameter | Sloppy; adds length without information | **Yes.** Cut it to one Methods section and put the hyperparameter heatmaps in supplementary material |
| R12 | Positive class never defined; confusion matrices have no row/column labels | Metrics can't be interpreted | **Yes.** Define positive = collapse-type; label every matrix |
| R13 | "Data will be made public in the future" | Reviewers want it now | **Yes.** Release it with a data card (Zenodo/IEEE DataPort) at submission |

---

## 2. What we can extract from the existing resources (no new falls needed)

### 2.1 Re-open the raw GetSensorData logs (most important first step)
GetSensorData (UAH/LOPSI) normally logs **all** phone sensors into one text file, with
line prefixes such as `ACCE`, `GYRO`, `MAGN`, `PRES`, `AHRS`, `SOUN`, `LIGH`.
The paper only used accelerometer magnitude. **Check the raw files for:**
- `GYRO`: angular velocity. This is the strongest possible addition: rotation during descent, and protective arm motion in pushes.
- `PRES`: barometer. Gives height change during the fall (Δh ≈ 0.5–1 m).
- `AHRS`/`MAGN`: orientation. Gives trunk/arm tilt before and after the fall.
- Timestamps: check the true sampling rate and jitter (Android sampling is not exactly 50 Hz). This is a sensor-quality result worth reporting.

If gyro/baro data exist, the paper moves from "accelerometer magnitude + ML" to a
**multi-sensor fusion study, using existing data**.

### 2.2 Recover metadata for every trial
Recover these from file names, timestamps, and notes/videos:

| Field | Needed for |
|---|---|
| `subject_id` (S1/S2) | LOSO evaluation, subject-confound test |
| `fall_class` (collapse/push) | label |
| `direction` (fwd/back/left/right/oblique) | leave-one-direction-out; confound test |
| `session/date`, `trial_order` | session leakage control, fatigue/learning effects |
| `phone model` | device heterogeneity |
| `push type` (aware/unaware/"fight") | sub-analysis |

If direction can't be recovered from notes, we can **estimate it from the gravity-vector
change** (pre-fall vs. post-fall orientation). That estimate becomes a contribution of its own.

### 2.3 Public datasets that contain the same *causal* categories
We do not need our own new falls if we use public data as **external validation**
of the two broader mechanisms: *collapse* (fainting) vs. *external perturbation* (slip/trip/push).

| Dataset | Relevant content | Placement | Use |
|---|---|---|---|
| **SisFall** (Sucerquia 2017) | Falls explicitly labelled "caused by fainting" (standing & sitting), plus slip/trip falls; 38 subjects incl. 15 elderly (ADLs only for most elderly) | Waist | External test: collapse vs. perturbation (slip/trip) |
| **KFall** (Yu et al. 2021) | SisFall-style protocol, 32 subjects, IMU incl. gyro, fall-onset labels | Low back | Same, with gyro; onset labels support segmentation |
| **UCI "Simulated Falls & ADL"** (Özdemir & Barshan 2014) | Includes "syncope" and "syncope-wall" falls among 20 fall types; 17 subjects, 6 body locations | Multi (incl. arms) | Placement study; closest match to our upper-arm position |
| FallAllD, UMAFall, UP-Fall | Mixed fall types, several placements | Varied | Pre-training / domain-shift analysis |

*(Verify the exact activity codes and licences before using them; the table above is from memory.)*

Honest framing: public data use waist/back placement and slip/trip rather than push. So
these experiments test whether **the discriminating mechanism** (a passive collapse vs. a
perturbation followed by a reactive response) transfers. They do **not** show that our exact
classifier transfers. That is still a much stronger paper than N = 2 alone.

---

## 3. Re-experiment plan

All experiments go through one scripted pipeline (`experiments/`), with fixed seeds and
every result saved to CSV. No step is done by hand.

### E0. Reproduce the old baseline (1 week)
- Rebuild the 121-sample magnitude windows from the raw files, then re-run the 4 original models under the **old** split.
- Purpose: show reviewers that the old numbers are reproducible, then show how they change under a proper protocol. Our own pipeline shows the inflation, which builds credibility.

### E1. Evaluation protocol (the core fix)
| Protocol | What it answers | Report |
|---|---|---|
| **Nested repeated stratified 5×5 CV** (10 repeats), grouped by session if known | Within-subject upper bound | mean ± SD, 95% bootstrap CI |
| **LOSO**: train S1 → test S2, and S2 → S1 | Does it generalize to an unseen person? | per-direction accuracy, balanced accuracy, MCC |
| **Leave-one-direction-out** | Is the model learning fall *direction* instead of *cause*? | accuracy per held-out direction |
| **Label-permutation test** (1000 perms) | Is performance above chance? | p-value |
| **Learning curve** (10…100% of train) | Would more data help? Supports the "pilot" claim | curve with CI |
| **Confound tests**: can the same features predict *subject ID* or *direction*? | Shortcut learning | accuracy of the confound classifiers |

Metrics: balanced accuracy, macro-F1, MCC, AUROC, sensitivity/specificity with a
defined positive class, and calibration (Brier score). Pairwise model comparison:
corrected resampled t-test (Nadeau–Bengio) or McNemar on LOSO predictions.

**Expected outcome:** within-subject CV stays high and LOSO drops. **We report both.**
Reviewers accept an honest cross-subject drop together with an analysis of why it drops. They reject
a single 92% number from a random split.

### E2. Automatic, impact-anchored segmentation (replaces visual inspection)
1. Detect impact as the maximum of the |a| peak after the free-fall dip (|a| < 0.6 g), using thresholds applied to *all* trials the same way.
2. Window = [impact − T_pre, impact + T_post]; sweep T_pre ∈ {0.5, 1, 1.5, 2 s} and T_post ∈ {0.5, 1, 2 s}.
3. Report segmentation success rate vs. the old manual windows, and classification sensitivity to alignment jitter (±100/200 ms).

### E3. Physically meaningful, phase-based features
Split each fall into phases: **pre-fall (prodrome / perturbation onset) → descent → impact → post-impact rest**.

| Phase | Collapse-type hypothesis | Push hypothesis | Features |
|---|---|---|---|
| Pre-fall | Slow sway, low-frequency motion, arm raised to head | Quiet, then a sudden jerk at push onset | RMS, dominant freq, jerk peak, arm-tilt change |
| Descent | Longer, "folding" descent; lower peak jerk | Short, abrupt; rotation about the push axis | descent duration, min |a| (free-fall depth), time-to-impact, gyro peak |
| Impact | Lower, broader peak (soft collapse) | Higher, sharper peak; arm used protectively | peak |a|, peak width, impulse, number of peaks |
| Post | Immobile (mimicked unconsciousness) | Early movement / attempt to get up | post-impact variance, time-to-first-movement, final orientation |

Feature sets compared: (F0) raw magnitude [old]; (F1) phase features on magnitude;
(F2) F1 plus tri-axial/orientation features; (F3) F2 plus gyro/baro if available.
Interpret with **SHAP / permutation importance**. The discussion then says *which
biomechanical phase* discriminates the classes, which is a scientific finding rather than just a leaderboard.

### E4. Models suited to small data
- Keep: SVM, GB, k-NN, MLP (as baselines).
- Add: **Random Forest / Logistic Regression on F1–F3** (interpretable).
- Add: **MiniRocket / ROCKET** (strong on small time-series data, cheap to compute).
- Add: a small 1D-CNN **with augmentation inside training folds only** (jitter, scaling, magnitude/time warp, 3-D rotation).
- Deep learning is *not* the headline. With 200 trials the message is that interpretable features ≈ or > DL.

### E5. Sensor-configuration study (this is what makes it an IEEE *Sensors* paper)
Simulate lower-cost or lower-power sensors by degrading the existing signals:

| Ablation | Levels | Sensor-design question |
|---|---|---|
| Sampling rate | 50 → 25, 12.5, 10, 5 Hz (anti-aliased resampling) | Minimum ODR for fall-cause, i.e., power budget |
| Measurement range | clip at ±2 g, ±4 g, ±8 g, full | Does impact saturation destroy cause information? (Many wearables default to ±2/4 g) |
| Resolution | 8-, 10-, 12-bit quantization | Can low-cost MEMS do it? |
| Noise | added white noise at several noise densities (µg/√Hz) | Robustness vs. MEMS noise specs |
| Axes | magnitude only / 3-axis / + gyro / + baro | Value of each modality |
| Window/latency | shortest post-impact window that keeps performance | Alert latency |
| Placement (public data) | upper arm (ours) vs. waist/wrist (UCI/SisFall) | Where to wear it |

Add an **on-device cost** table: feature-extraction + inference time, model size (KB),
measured on a phone or a Cortex-M-class board / Raspberry Pi. Output: a
*"minimum sensor specification for fall-cause recognition"* figure. That figure is the paper's
distinctive sensor contribution.

### E6. External validation on public data (Sec. 2.3)
1. Relabel public falls into **collapse (fainting)** vs. **perturbation (slip/trip)**; drop ambiguous types.
2. Within public data: LOSO across 15–38 subjects (this answers the N = 2 criticism for the *mechanism*).
3. Cross-dataset: train on public → test on ours, and ours → public, using orientation-invariant features (magnitude/phase features, resampled to a common rate).
4. Report the domain-shift gap and what closes it (feature normalisation, rate matching).

### E7. Robustness and failure analysis
- Show which trials are misclassified (by direction and subject), with signal plots.
- Report sensitivity to the segmentation threshold and to phone model.

---

## 4. Re-framed paper

**Working title (Journal):**
*"Collapse or Perturbation? Phase-Aware Wearable Accelerometer Sensing for Fall-Cause
Recognition, with Sensor-Configuration Analysis and Cross-Dataset Validation"*

**Contributions (each maps to an experiment):**
1. Formulation of fall-**cause** recognition (collapse vs. external perturbation) as a sensing problem distinct from fall detection. (Intro, related work)
2. An openly released pilot upper-arm smartphone dataset (200 trials) with a data card and full metadata. (Sec. 2)
3. Automatic, impact-anchored phase segmentation and physically interpretable features, showing *which* fall phase carries cause information. (E2, E3)
4. A small-sample evaluation protocol (nested CV, LOSO, permutation, confound tests) with honest cross-subject results. (E1)
5. Sensor-specification analysis: sampling rate, range, resolution, noise, modality, placement, on-device cost. (E5)
6. External validation of the collapse-vs-perturbation mechanism on public multi-subject datasets. (E6)

**Section outline (IEEE Sensors J., ~10–12 pages):**
I Introduction → II Related work (fall detection vs. fall *characterisation*: direction,
severity, syncope. Cite Nyan et al., Syed et al. [14], SisFall/KFall) → III Sensing setup & dataset
(incl. ethics, data card) → IV Signal processing (segmentation, phases, features) →
V Evaluation protocol → VI Results (E1–E6) → VII Discussion (biomechanics, limitations,
simulated vs. real syncope, N = 2) → VIII Conclusion.

**Venue choice:**

| Venue | Fit | Recommendation |
|---|---|---|
| **IEEE Sensors Letters** (4 pages) | Pilot studies with a clear sensor finding are acceptable | **Safest target** if E6 (public data) is weak or unavailable: E1 + E3 + E5 in 4 pages |
| **IEEE Sensors Journal** | Needs the sensor contribution (E5) **and** some multi-subject evidence (E6) | Go for it if E6 gives reasonable results |
| IEEE Sensors Conference | Short and fast | Fallback / an early version of the Letter |

---

## 5. Do's and don'ts for the rewrite

**Do**
- Say "pilot", "simulated", and "two participants" in the abstract. Reviewers are more lenient when a limitation is stated up front and more severe when they have to find it.
- Report LOSO even if it is low. Explain it using the confound and SHAP analyses.
- Release the code and data. Put the hyperparameter grids and heatmaps in the supplement.
- Add the ethics approval number and informed-consent statement.

**Don't**
- Don't compare our accuracy with fall-*detection* accuracies from other papers.
- Don't claim forensic or legal use (murder/assault scenario).
- Don't select models or hyperparameters on the test set, and don't report only the best split.
- Don't augment or oversample before splitting the data.

---

## 6. Timeline (≈ 8–10 weeks)

| Week | Work |
|---|---|
| 1 | Raw-log audit (2.1), metadata recovery (2.2), ethics paperwork check, E0 reproduction |
| 2–3 | Auto segmentation (E2), phase features (E3), evaluation harness (E1) |
| 4 | Models (E4), confound and permutation tests |
| 5 | Sensor ablations + on-device timing (E5) |
| 6–7 | Public datasets: download, relabel, LOSO and cross-dataset (E6) |
| 8 | Failure analysis (E7), figures, choose Letters vs. Journal based on E6 |
| 9–10 | Rewrite, data card, Zenodo release, internal mock review against the table in Sec. 1 |

## 7. Decision points
- **After week 1:** if gyro/baro logs exist, fusion becomes a headline contribution.
- **After week 4:** if LOSO ≈ chance *and* confound tests show subject leakage, pivot the story to *"phase features that transfer vs. those that don't"* and target Sensors Letters.
- **After week 7:** if public-data LOSO for collapse vs. perturbation is good (> 80% balanced accuracy), target Sensors Journal.
