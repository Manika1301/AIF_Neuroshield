# What NeuroShield predicts — the definitive spec

This is the contract between the model and the product: exactly what goes in, exactly what comes
out, and exactly what each output is allowed to claim. Every number here is sourced from the code,
not from memory. If code and this document disagree, the code is right and this is a bug.

Unit of prediction: **one 60-second window, stepped every 30 seconds** (`extract.py:57-58`).
Everything below is produced per window.

---

## 1. Inputs — 36 features per window

### 1a. The 19 absolute features (`features-v2`, `features/extract.py:22-44`)

Derived from four Empatica E4 wrist channels: BVP/PPG (64 Hz), EDA (4 Hz), TEMP (4 Hz), ACC (32 Hz).

| Group | Columns | What it measures |
|---|---|---|
| Pulse | `hr_mean_bpm`, `ibi_sd_ms`, `ibi_rmssd_ms` | Heart rate and beat-to-beat variability. Variability *drops* under stress. |
| Pulse quality | `ppg_quality` | How trustworthy the pulse signal is. Gates abstention; never explains a status. |
| Frequency-domain HRV | `hrv_lf`, `hrv_hf`, `hrv_lf_hf_ratio` | Sympathovagal balance. A higher LF/HF ratio means more sympathetic ("fight-or-flight") dominance. |
| Skin conductance | `eda_level`, `eda_slope`, `eda_response_count`, `eda_response_mean_amp` | Sweat-gland activity — the most direct peripheral read on arousal. |
| EDA decomposition (cvxEDA) | `eda_tonic_mean`, `eda_tonic_slope`, `eda_phasic_mean` | Splits *sustained* arousal (tonic) from *momentary* responses (phasic). |
| Skin temperature | `temp_mean_c`, `temp_slope_c_per_min` | Peripheral vasoconstriction: skin gets **cooler** under stress. |
| Motion | `motion_dynamic_rms`, `motion_dynamic_p95` | Wrist movement. Used to abstain, and reported as its own axis. |
| Coverage | `valid_fraction` | Fraction of the window with usable samples. Gates abstention. |

A feature that cannot be computed for a window is `NaN`, never silently dropped. The
gradient-boosting heads consume NaN natively.

### 1b. The 17 personalized features (`features/personalize.py`)

Every physiological feature above is **restated as a deviation from that user's own quiet
baseline**: `(x − person_mean) / person_std`, suffix `_p`.

The reference is the user's **first 300 seconds of accepted windows** — the same short quiet
calibration the live app performs, so training and serving compute this identically. The reference
is defined in *seconds*, deliberately: a window *count* would silently mean a different calibration
duration at a different window step.

`ppg_quality` and `valid_fraction` are **not** personalized. They describe the recording, not the
person; "this recording is 12% noisier than your usual recording" is not a physiological statement.

**Why this exists:** two people can both be perfectly calm at 58 and 82 bpm. A model fed only
absolute units burns capacity learning *who someone is*, and cannot do it at all for a person it has
never seen. Adding the personalized half moved WESAD accuracy from 0.880 to 0.919. Both halves are
kept, because absolute level still carries information (very high EDA is unusual for *anyone*).

---

## 2. Outputs

### 2a. Head A — graded stress (the headline)

Calibrated `HistGradientBoostingClassifier` (isotonic), trained on **WESAD only**.

| Field | Type | Definition |
|---|---|---|
| `stress_prob` | float 0–1 | Calibrated P(stress). Isotonic calibration is what makes this a real probability rather than an arbitrary score. |
| `stress_index` | **int 0–100** | `round(stress_prob × 100)`. The product's hero KPI. |
| `level` | `calm` / `elevated` / `high` | `calm` < 0.45 ≤ `elevated` < 0.70 ≤ `high` (`artifact.py:36`). |

### 2b. Head B — affect state

4-class `HistGradientBoostingClassifier`, trained on WESAD's affect labels (the only dataset that
has them).

| Field | Type | Definition |
|---|---|---|
| `affect_state` | `baseline` / `stress` / `amusement` / `meditation` | Argmax class. |
| `affect_confidence` | float 0–1 | Max class probability. |

**Why a second head:** arousal alone cannot tell excitement from distress. A raised heart rate during
something *enjoyable* (amusement) looks a lot like stress to a single binary model. Head B is what
lets the product avoid calling a good moment a bad one.

### 2c. The four physiological axes — **not model outputs**

`cardiac`, `electrodermal`, `thermal`, `movement`, each `{score, level, n_features}`
(`runtime/axes.py`).

These are **direct, honest transforms of the user's own z-scores** — no model, no learning. Each axis
averages its features' z-scores *in the arousal direction*, which is not always "higher = worse":
pulse variability and skin temperature carry a **−1** sign, because falling variability and cooling
skin both indicate *more* arousal.

`score` is clipped to ±5. `level` is `high` at ≥2.0, `elevated` at ≥1.0, else `normal`. An axis with
no usable features reports `score: null`, not a fake zero.

Their purpose is explanatory: they answer "*which system* is driving this number", which a single
0–100 index cannot.

### 2d. Status state — one of 9

`waiting`, `calibrating`, `green`, `amber`, `red`, `motion_paused`, `poor_signal`, `stale`, `error`.

Resolution order in `StatusStateMachine.update()` — **abstention outranks the model**:

1. `stale` — no data for >120 s.
2. `calibrating` — no personal baseline yet.
3. `motion_paused` / `poor_signal` — **abstention (see 2e)**.
4. `waiting` — no probability available.
5. `green` / `amber` / `red` — the model's band, with **2-window hysteresis** so the light cannot
   flicker on a single noisy window.

### 2e. Abstention — when we refuse to predict

Checked *before* the model, in `runtime/quality_gate.py`. On abstention the engine **does not call
the model at all**: `stress_prob`, `stress_index`, `level`, and `affect_state` are all `null` for
that window.

| Reason | Trigger |
|---|---|
| `motion_paused` | `motion_dynamic_rms` > 1.0 m/s² **or** `motion_dynamic_p95` > 2.0 m/s² |
| `poor_signal` | `valid_fraction` < 0.9, **or** `ppg_quality` < 0.7, **or** motion unmeasurable |

Motion is checked first, so a motion-caused quality collapse is reported as motion — the reason
names the actual cause. **Refusing to answer is a feature, not a gap:** a wrist PPG during hand
movement is not a stress signal, and a model that guesses anyway is worse than one that says "I
can't see right now."

### 2f. Reasons — plain language, never diagnosis

0–3 sentences per colored status, at most one per group (pulse / EDA / temperature), ranked by |z|
(`runtime/explain.py`). Fixed templates, e.g. *"Skin-response activity is above your quiet
baseline."*

`assert_no_clinical_claims()` hard-blocks nine forbidden terms (panic, anxiety, burnout, diagnos-,
clinical, medical, …) and raises rather than emit them. This is a regression guard, not a style
guide — see `docs/no_clinical_claims.md`.

### 2g. Session-level (Tier-3, `runtime/dynamics.py`)

| Field | Meaning |
|---|---|
| `time_in_state` | Seconds spent in each of the 9 states |
| `recovery_trend` | `rising` / `falling` / `steady` — slope of the last 5 stress indices |
| `episodes` | Contiguous amber/red runs ≥2 windows: `{start_s, end_s, n_windows, peak_index, peak_state}` |
| `hrv_proxy_recovery` | Mean `ibi_rmssd_ms` over the last 5 windows |
| `index_summary` | `{mean, max, latest}` stress index |

---

## 3. What these predictions may and may not claim

**Accuracy.** Head A: **0.919** balanced accuracy on WESAD under grouped leave-one-subject-out —
i.e. measured only on people the model never trained on. Head B: **0.616** (4-class, vs a 0.250
chance baseline). Dummy baseline for Head A is 0.500.

**The honest error bar.** With 15 subjects the standard error is **±0.036**, and per-subject accuracy
ranges from **0.50 on the worst subject to 1.00 on the best**. This model is *not* uniformly good; it
is good on average and can be near-useless on an individual. Do not present the 0–100 index as a
precise measurement of a specific person.

**Not yet measured for this model:** cross-dataset generalization. Stress-Predict and the Nurse
Stress shifts are held out and have **not** been re-scored against `m3`. The scoreboard on disk is
from a superseded model and is marked stale. Nothing in the product should quote a cross-dataset
number until `scripts/build_scoreboard.py` is re-run.

**Not a clinical device.** This does not detect, diagnose, or predict panic attacks, anxiety
disorders, burnout, or any medical condition. It reports *physiological arousal* relative to a
person's own baseline, and says so in those words. See `docs/no_clinical_claims.md`.

**Trained on lab data, mostly.** WESAD is a controlled protocol (TSST stress induction). Head B's
four affect states come from that one protocol. Real hospital shifts are messier than anything the
model was trained on.
