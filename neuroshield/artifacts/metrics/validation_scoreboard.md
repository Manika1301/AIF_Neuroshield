> **STALE — DO NOT CITE.** These numbers come from the old pooled `m2` model, which no longer
> exists. They predate `features-v2`, per-subject personalization, and the WESAD-only training
> decision. The current WESAD figure is **0.919** (see `docs/model_card_m3_multihead.md`).
> Regenerate with: `uv run python scripts/build_scoreboard.py`

# Three-dataset validation scoreboard (Head A: graded stress)

Generated: 2026-07-11T19:31:29.515357+00:00
Task: binary graded stress (baseline vs. stress)

| Dataset | Evaluation | Balanced acc. | Macro F1 | Windows |
|---|---|---|---|---|
| nurse_stress | held-out (frozen model, never trained on) | 0.561 | 0.384 | 7491 |
| stress_predict | grouped-LOSO (in training pool) | 0.588 | 0.583 | 3710 |
| wesad | grouped-LOSO (in training pool) | 0.725 | 0.741 | 919 |

## Note

Train-pool datasets (WESAD, Stress-Predict) are scored by grouped leave-one-subject-out, not by predicting their own training rows. Nurse Stress is genuinely held out. Naturalistic Nurse numbers are expected to be lower than lab-protocol datasets: labels are sparse self-report and the setting is uncontrolled real hospital work.