# Datasets

Status of every dataset named in the project's task plan, and where things live on disk.
Raw/external data is never committed (`data/raw/`, `data/external/`, `data/interim/` are
git-ignored); only small synthetic fixtures under `data/fixtures/` are committed.

## WESAD -- mandatory, used

**Status: downloaded, extracted, used for M1 training (T4-T9).**

- Source: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
  (redirects to the original University of Siegen page, direct file via a public Sciebo share).
- Location: `data/external/wesad/` (subject folders `S2`-`S17`, skipping `S1`/`S12`, per the
  official release). Provenance recorded in `data/external/wesad/SOURCE.txt` (URL, date, size,
  checksums, license, citation).
- Used for: `src/neuroshield/data/wesad_loader.py`, the `features-v1` extractor, and the frozen
  M1 model (`artifacts/models/m1_wesad_features_v1.joblib`). LOSO evaluation: balanced accuracy
  0.831, macro-F1 0.822 (`artifacts/metrics/m1_loso_summary.json`).

## Stress-Predict Dataset -- external validation, used

**Status: cloned, used for external validation (T18).**

- Source: https://github.com/italha-d/Stress-Predict-Dataset (git clone).
- Location: `data/external/stress_predict/`. Provenance in
  `data/external/stress_predict/SOURCE.txt` (commit hash, date, citation).
- Used for: `src/neuroshield/data/stress_predict_loader.py` and
  `src/neuroshield/models/external_validation.py`. Same Empatica E4 wrist channels as WESAD, so
  every `features-v1` column is computable. Result: balanced accuracy 0.541, macro-F1 0.506 across
  34 labelled subjects (`artifacts/metrics/external_validation_notes.md`) -- modest but
  above-chance cross-dataset, cross-stressor-protocol generalization, as expected.

## PPG-DaLiA -- strongly recommended, skipped (documented)

**Status: not downloaded. Skip decision: deferred to O1 (optional task), not required for the
core software-first MVP.**

PPG-DaLiA (UCI, ~2.7 GB) exists to replace the currently-guessed motion/quality thresholds in
`src/neuroshield/runtime/quality_gate.py` with thresholds derived from real heart-rate-error-vs-
motion data. The MVP's thresholds were instead calibrated against this project's own synthetic
motion-burst phase (see the module's docstring) with a wide safety margin, and validated
end-to-end via the T13/T14 test suites and the T19 software acceptance run. This is an accepted,
documented simplification for the MVP gate -- O1 (`tasks.tex` Section 4, Optional Tasks) is the
follow-up to replace it with evidence from real data, and requires no changes to the surrounding
architecture (`quality_gate.py`'s constants are the only thing that would change).

## Nurse Stress Dataset -- naturalistic validation + Tier-4 insights, used

**Status: downloaded, used for held-out real-world validation and descriptive context analytics.**

- Source: https://datadryad.org/dataset/doi:10.5061/dryad.5hqbzkh6f (Dryad,
  doi:10.5061/dryad.5hqbzkh6f). Provenance in `data/external/nurse_stress/SOURCE.txt`; the raw
  `Stress_dataset.zip` (~1.16 GB) stays at its download location and is read in memory, never
  re-extracted (see `nurse_stress_loader.py`).
- Location: `data/doi_10_5061_dryad_5hqbzkh6f__v20210917/`.
- Used for: (1) **held-out** real-world validation of the multi-head model -- never in training,
  because its 0/1/2 labels are sparse self-report from real hospital shifts (see
  `docs/build_roadmap.md` and `docs/design_doc.tex`); (2) **Tier-4 descriptive** context analytics
  (`models/nurse_insights.py` -> `artifacts/metrics/nurse_context_insights.md`): which survey
  triggers co-occur with high self-reported stress (e.g. "treating a COVID patient" lift 2.06),
  explicitly descriptive and never a live cause predictor.
- 15 nurses, ~620 recording sessions, 225 labelled events (179 stress / 46 baseline). Survey times
  aligned to UTC via `America/New_York` (verified by cross-referencing session start times).

## SWELL-KW -- optional (M2 cognitive-load module), skipped (documented)

**Status: not requested/downloaded. Skip decision: out of scope for M1; this dataset only feeds
a separate, explicitly experimental M2 classifier (task plan O2), which the core software-first
MVP does not require.**

Access reportedly requires a request rather than an immediate direct download
(http://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html); not pursued since M2 is optional.

## Fitbit Fitness Tracker Data -- optional (trends screen), skipped (documented)

**Status: not downloaded. Skip decision: optional trends/history UI (task plan O3), not part of
the M1 stress-proxy MVP or its acceptance gate.**

Kaggle dataset (`arashnic/fitbit`) with no stress/burnout labels; would only support a
"descriptive trend" dashboard screen (explicitly not framed as burnout prediction, per
`docs/no_clinical_claims.md`), which the software acceptance gate (T19) does not require.
