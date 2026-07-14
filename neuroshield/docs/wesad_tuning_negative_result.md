# WESAD tuning: a negative result

**Summary: WESAD Head A is at 0.919 LOSO balanced accuracy, and that is the practical ceiling for
this dataset and feature set. A systematic tuning sweep found nothing that beats it.** This file
exists so nobody spends another day re-running these experiments.

## What is real

Two research-backed changes took Head A from **0.831 → 0.919**, and both are large enough to clear
the noise floor:

| Change | Head A LOSO bal. acc. |
|---|---|
| M1: 13 absolute features, logistic regression | 0.831 |
| + `features-v2` (frequency-domain HRV, cvxEDA tonic/phasic) | 0.880 |
| + per-subject baseline personalization (**shipped, `m3`**) | **0.919** |

## What is not real

Everything below was tried and **none of it beats 0.919**. Levers swept: window density (10s vs 30s
step), rolling-median prediction smoothing (k = 2/3/5/9), seed ensembling (5 seeds), boosting
hyperparameters (depth, L2, learning rate, iterations), `class_weight="balanced"`, in-fold decision
threshold tuning, a robust median/IQR personal reference, and calibration-reference duration
(120s/150s/300s/600s).

Scored per held-out subject and compared to the shipped config with a paired Wilcoxon signed-rank
test on the same 15 subjects (`scripts/wesad_significance.py`):

| Config | Mean per-subject bal. acc. | vs. shipped | p |
|---|---|---|---|
| **shipped: step 30s / ref 300s** | 0.918 ± 0.036 | — | — |
| step 10s / ref 120s | 0.917 ± 0.034 | −0.000 | 0.58 |
| step 10s / ref 300s | 0.894 ± 0.040 | −0.024 | 0.18 |
| step 30s / ref 150s | 0.921 ± 0.037 | +0.003 | 0.92 |
| step 10s / ref 150s + 5-seed + strong L2 | 0.915 ± 0.033 | −0.003 | 0.67 |

**Not one is distinguishable from noise.** Two findings are worth keeping:

- **Prediction smoothing actively hurts** (0.927 → 0.922 → 0.915 → 0.903 as k grows). It blurs the
  stress onset/offset transitions. This is a lever that sounds obviously good and is not.
- **A robust median/IQR personal reference is worse** than plain mean/std (0.900 vs 0.927).

## Why the ceiling is the data, not the model

The standard error of a LOSO estimate over 15 subjects is **±0.036**, so nothing smaller than a
~7-point gap is detectable here. Per-subject balanced accuracy for the shipped config ranges from
**0.50 on the worst subject to 1.00 on the best** — that between-person spread *is* the error bar.

Getting meaningfully past 0.92 needs more subjects or better labels, not more hyperparameter search.

## Two traps this exercise fell into (both caught, both instructive)

**Selection on the test set.** Scoring 16 configs against the same 15 LOSO folds and reporting the
best produced an apparent 0.940. That configuration scores 0.917 — i.e. the shipped number — once
the reference bug below is fixed. If you sweep, the number you quote must come from nested CV
(`--nested`) or a paired significance test, never from the max of the sweep.

**A hyperparameter hidden inside a bug.** The personal-baseline reference was originally defined as
a count of windows (10). At a 30s step that is 5 minutes; at a 10s step it silently becomes
2.5 minutes. So changing the window step also changed the calibration period, and the "denser
windows help" result was partly that accident. The reference is now defined in **seconds**
(`DEFAULT_REFERENCE_SECONDS = 300.0`), which is also what keeps training aligned with the
fixed-duration calibration the live app performs. A non-monotonic sweep (120s → 0.919, 150s → 0.895,
300s → 0.896, 600s → 0.914) is what exposed it: a real effect trends, noise bounces.

## Reproducing

```bash
uv run python scripts/cache_wesad_features.py        # extract once (slow: cvxEDA)
uv run python scripts/wesad_experiments.py --round 3 # sweep
uv run python scripts/wesad_experiments.py --round nested --nested  # unbiased estimate
PYTHONPATH=scripts uv run python scripts/wesad_significance.py      # paired per-subject test
```
