# Honesty boundary: no clinical claims

NeuroShield is **not** a clinical device, and nothing in this repository, its models, its
dashboard copy, or its documentation should claim otherwise.

## What the MVP actually does

The MVP detects patterns associated with public stress-proxy datasets (for example WESAD) and
with local simulated/replayed sensor streams. The labels used for training and evaluation are
proxy labels defined by the dataset authors' lab protocols, not clinical diagnoses. Examples of
these proxy labels include:

- Baseline / neutral / quiet rest periods.
- Lab-induced stress (e.g. Stroop test, public speaking, cold pressor, timed tasks).
- Amusement or relaxation control conditions.
- General activity or motion periods.
- Self-reported stress or mood surveys.

A model trained on these labels learns to recognize physiological patterns statistically
associated with these proxy conditions in the study population. It does not learn to recognize a
clinical event in an individual user.

## What the MVP must never claim

The product, its status labels, its explanations, its dashboard, and any generated report must
never state or imply that the system:

- Predicts or detects a panic attack.
- Diagnoses anxiety or any anxiety disorder.
- Diagnoses heat illness or any medical condition.
- Detects or predicts burnout.
- Provides a clinical or medical assessment of any kind.

## How this constrains implementation

- Status states (`green`/`amber`/`red`, `motion_paused`, `poor_signal`, etc.) describe model
  output relative to a personal baseline, not a medical state.
- Explanations must be restrained and feature-based (e.g. "skin-response activity is above your
  quiet baseline"), never diagnostic (e.g. never "panic attack starting").
- Any longitudinal/trend feature (e.g. built on Fitbit-style data) must be labeled as
  descriptive trend information only, never framed as burnout prediction.
- Model cards and documentation must state the proxy-label limitation explicitly wherever
  accuracy or performance is discussed.
