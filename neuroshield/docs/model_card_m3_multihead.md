# Model card: M3 multi-head, personalized (graded stress + affect)

## What this model does

M3 is the multi-output model behind the redesigned product (see `design_doc.tex`). Given a
60-second window it produces two things:

- **Head A -- graded stress:** a calibrated probability mapped to a **0-100 stress index** and an
  ordinal **calm / elevated / high** level.
- **Head B -- affect state:** a 4-class label -- **baseline / stress / amusement / meditation** --
  distinguishing stress-arousal from positive-arousal (amusement) and calm (meditation).

It is not a clinical or diagnostic tool. See `docs/no_clinical_claims.md`.

- **Model version:** `m3_multihead_personalized_v1`
- **Feature version:** `features-v2`
- **Artifact:** `artifacts/models/m3_multihead_personalized_v1.joblib` + `_manifest.json`
- **Base learner:** `HistGradientBoostingClassifier` (handles NaN natively); Head A wrapped in
  `CalibratedClassifierCV` (isotonic) so the index is a true probability.

## Input features (36)

Each window is described twice -- in absolute units, and relative to that person's own quiet state.

**19 absolute `features-v2` columns** (`features/extract.py`):

| Group | Columns |
|---|---|
| Pulse (PPG/BVP) | `hr_mean_bpm`, `ibi_sd_ms`, `ibi_rmssd_ms`, `ppg_quality` |
| Frequency-domain HRV | `hrv_lf`, `hrv_hf`, `hrv_lf_hf_ratio` |
| Skin conductance (EDA) | `eda_level`, `eda_slope`, `eda_response_count`, `eda_response_mean_amp` |
| EDA tonic/phasic (cvxEDA) | `eda_tonic_mean`, `eda_tonic_slope`, `eda_phasic_mean` |
| Skin temperature | `temp_mean_c`, `temp_slope_c_per_min` |
| Motion (ACC) | `motion_dynamic_rms`, `motion_dynamic_p95` |
| Coverage | `valid_fraction` |

**17 personalized `_p` columns** (`features/personalize.py`): every physiological feature above,
re-expressed as a deviation from that subject's own baseline (`(x - person_mean) / person_std`).
The reference is the subject's **first 300 seconds of accepted windows** -- the same short quiet
calibration the live app already performs, so the served features are computed exactly as the trained
ones were. (The reference is deliberately defined in *seconds*, not in a count of windows: a window
count would silently mean a different calibration duration at a different window step, and train and
serve would drift apart.) The two quality/coverage columns are not personalized -- they describe the
recording, not the person.

This personalization is the single biggest accuracy lever in the model, and it is what the
literature predicts: two people can both be calm at 58 and 82 bpm, so a model fed only absolute
units wastes capacity learning *who someone is* -- and cannot do that at all for a person, or a
dataset, it has never seen.

## Training data

- **Head A:** WESAD (baseline vs. stress) only, grouped LOSO by subject. Stress-Predict and
  Nurse Stress are **held out** (never trained on). Naive pooling with Stress-Predict was tried
  and *lowered* accuracy (0.88 -> 0.62), matching the literature; it was dropped.
- **Head B:** WESAD only -- it is the only dataset with amusement/meditation labels (which the
  original binary pipeline discarded).

## Evaluation (grouped leave-one-subject-out on WESAD)

| Head | Task | Model bal. acc. | Dummy bal. acc. |
|---|---|---|---|
| A | binary graded stress | **0.919** | 0.500 |
| B | affect 4-class | **0.616** | 0.250 |

Full metrics: `artifacts/metrics/m3_multihead_loso.json`.

Cross-dataset generalization (Stress-Predict and Nurse Stress as genuinely held-out rows) has **not
yet been re-measured for this model** -- `artifacts/metrics/validation_scoreboard.md` still holds the
old pooled-`m2` numbers and is marked stale. Regenerate it with
`uv run python scripts/build_scoreboard.py` before quoting any cross-dataset figure.

### How Head A got here

Every number below is grouped LOSO on WESAD -- the same subjects, the same protocol, the same
metric, so the steps are directly comparable.

| Step | Head A bal. acc. |
|---|---|
| M1: 13 absolute features, logistic regression, WESAD only | 0.831 |
| M2: pooled WESAD + Stress-Predict training, 13 features | 0.617 |
| M2: drop naive pooling (WESAD-only training) | 0.831 |
| + `features-v2` (frequency-domain HRV, cvxEDA tonic/phasic) | 0.880 |
| + per-subject baseline personalization (**M3**) | **0.919** |

A systematic tuning sweep on top of this (window density, prediction smoothing, seed ensembling,
calibration length, boosting hyperparameters, threshold tuning) produced **no further gain that
survives a paired per-subject significance test** -- 0.919 is the practical ceiling for this dataset
and feature set. The dead ends are written up in `docs/wesad_tuning_negative_result.md` so they are
not re-run. With 15 subjects the standard error on a LOSO estimate is +/-0.036, so only differences
larger than ~0.07 are detectable at all.

For context, published wrist-only stress detection evaluated with the same LOSO discipline tops out
around 0.87 (Siirtola 2019). Papers reporting 0.93-0.99 almost always use k-fold or intra-subject
splits, which let the same person appear in train and test -- an evaluation we deliberately do not
use, because it does not answer "will this work on a new nurse."

## Intended use / limitations

- Drives the live 0-100 index, level, affect chip, and the four physiological axes in the dashboard,
  after personal baseline calibration and the motion/quality abstention gate.
- **Requires a calibration period.** A user who skips it gets no personalized features. The model
  still runs (the gradient-boosting heads handle the resulting NaNs natively), but at the weaker
  absolute-features-only accuracy.
- Head B's affect distinction is trained on one lab protocol (WESAD) and should be read as
  suggestive, not definitive.
- Explanations use z-score ranking (the gradient-boosting base is not linear), so live reasons are
  feature-direction statements, never diagnoses.
- Manifest pins the exact `feature_version` and a checksum; loading refuses a mismatch rather than
  producing a misaligned prediction.
