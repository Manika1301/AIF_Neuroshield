# Model card: M1 (rest vs. stress-proxy)

## What this model does

M1 is a binary classifier: given a 60-second window of wrist-worn heart, skin-response,
temperature, and motion features, it estimates the probability that the window resembles a
**lab stress-proxy condition** rather than a **quiet baseline condition**, as those conditions
were defined by the WESAD study protocol. It does not predict panic attacks, diagnose anxiety, or
detect burnout -- see `docs/no_clinical_claims.md`.

- **Model type:** logistic regression (`SimpleImputer` → `StandardScaler` → `LogisticRegression`,
  `class_weight="balanced"`).
- **Model version:** `m1_wesad_features_v1`
- **Feature version:** `features-v1` (13 columns; see `src/neuroshield/features/extract.py`)
- **Artifact:** `artifacts/models/m1_wesad_features_v1.joblib` +
  `artifacts/models/m1_wesad_features_v1_manifest.json`

## Training data

- **Dataset:** WESAD (Wearable Stress and Affect Detection), 15 subjects, wrist Empatica E4
  channels only (BVP, EDA, TEMP, ACC).
- **Labels:** WESAD's own baseline (→ 0) and stress (→ 1) protocol segments. Amusement,
  meditation, transient/undefined periods, and unused label codes were excluded, not folded into
  either class (see `src/neuroshield/features/labels.py`).
- **Windows used for training:** 919 of 2,874 total extracted windows survived labelling and the
  quality gate (≥90% valid signal coverage). Class balance: 587 baseline, 332 stress.
- **What M1 has never seen:** any data outside WESAD's wrist channels, any subject held out
  during evaluation (see below), and the entire Stress-Predict dataset (used only for validation).

## Evaluation

### Within-dataset: leave-one-subject-out (LOSO) on WESAD

The only way to get an honest read on generalization to a new person: each of the 15 subjects is
held out as the test fold in turn, never contributing a single window to that fold's training
data (`src/neuroshield/models/train_m1.py`).

| Metric | M1 | Dummy baseline (most-frequent-class) |
|---|---|---|
| Balanced accuracy | **0.831** | 0.500 |
| Macro F1 | **0.822** | 0.390 |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[489, 98], [57, 275]]` | `[[587, 0], [332, 0]]` |

Full per-subject breakdown: `artifacts/metrics/m1_loso_summary.json`. Per-window predictions:
`artifacts/metrics/m1_loso_predictions.csv`.

### Cross-dataset: Stress-Predict (external validation, T18)

Run unchanged (frozen artifact, no retraining) against 34 Stress-Predict subjects with usable
ground truth, whose stress protocol (Stroop test, interview, hyperventilation) differs from
WESAD's:

| Metric | Value |
|---|---|
| Subjects evaluated | 34 |
| Windows evaluated | 3,710 |
| Balanced accuracy | 0.541 |
| Macro F1 | 0.506 |

This is modest but above chance, and that is expected and honestly reported: it is a harder
test of generalizing across *stress inducers*, not just across people. Details and per-subject
numbers: `artifacts/metrics/external_validation_notes.md`.

## Intended use

- Driving the software-first MVP's live status (green/amber/red) after personal baseline
  z-scoring and the motion/quality abstention gate, within this project only.
- Research and demonstration purposes on the datasets and synthetic/replay streams described in
  this repository.

## Out of scope / limitations

- **Not a clinical or diagnostic tool.** See `docs/no_clinical_claims.md` -- this applies to every
  use of M1's output, including in explanations and dashboard copy.
- **Trained on 15 people, all captured in one lab protocol.** LOSO accuracy (0.831) measures
  generalization to a new *person* under the *same* protocol; it does not measure generalization
  to a new stress *inducer*, a real-world setting, or hardware other than an Empatica E4-class
  wrist sensor -- the external validation numbers above are the honest evidence for that harder
  question, and they are meaningfully lower.
- **Class imbalance.** Baseline windows outnumber stress windows roughly 1.8:1 in training;
  `class_weight="balanced"` compensates during fitting, but this is worth keeping in mind when
  interpreting any single low-count subject's per-subject metrics.
- **Feature/model version coupling.** The manifest pins an exact `feature_version`; the loading
  code (`src/neuroshield/models/artifact.py`) refuses to run the model against a mismatched
  feature set rather than silently producing a wrong prediction.
