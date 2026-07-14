# NeuroShield build roadmap — beyond the traffic light

## Why this exists

The MVP produces a single binary stress probability → green/amber/red. We have three Empatica-E4
datasets (WESAD, Stress-Predict, Nurse Stress) and a rich, mostly-unused label space. This roadmap
turns the product into a **personalized, multi-dimensional stress & recovery application** without
overclaiming (see `docs/no_clinical_claims.md`).

Decisions locked with the user (2026-07-11):
- **Model:** multi-output (multiple trained heads + derived axes).
- **Frontend:** React/Next.js talking to the FastAPI backend (visual polish accepted as
  un-verifiable in the build environment; we verify build + API contract, not pixels).
- **Insights:** include Tier-4 descriptive nurse context analytics + a 3-dataset validation
  scoreboard.

## What we predict / show (the KPIs)

**Tier 1 — 4 physiological axes (derived, personalized z-scores, no model):**
cardiac arousal, electrodermal arousal, thermal response, movement/activity.

**Tier 2 — Multi-output model:**
- Head A — graded stress: 0–100 calibrated index + ordinal calm/elevated/high.
- Head B — affect state: 4-class baseline / stress / amusement / meditation (uses WESAD's
  amusement + meditation labels that T7 currently discards).

**Tier 3 — Session & recovery analytics (derived, no model):** recovery trend, time-in-state,
sustained stress episodes, HRV-proxy recovery score.

**Tier 4 — Research insights (offline, descriptive only):** nurse context/trigger vs. arousal
correlation table; 3-dataset validation scoreboard.

## Dataset responsibilities

| Dataset | Role |
|---|---|
| WESAD | Head A (baseline/stress) + Head B (4-class) training; LOSO evidence |
| Stress-Predict | Pooled into Head A training (clean binary labels); external validation |
| Nurse Stress | Held-out real-world validation + Tier-4 context insights (labels too sparse/self-reported to train on) |

Artifacts are versioned; the existing `m1_wesad_features_v1` is never overwritten.

## Phases

### Phase 1 — Data & ML core
- `nurse_stress_loader.py`: in-memory nested-zip reading, `America/New_York` survey label
  alignment (see the approved plan `~/.claude/plans/calm-squishing-dahl.md`).
- Label harmonization across the 3 datasets into a common schema.
- Multi-output training: Head A (pooled, calibrated) + Head B (WESAD 4-class), grouped CV by
  `(dataset, subject)`, honest metrics. Versioned artifact `m3_multihead_personalized_v1`.
- 4-axis decomposition module (derived scores).
- 3-dataset validation scoreboard + nurse context analytics.

### Phase 2 — Backend enrichment
- Enrich the status payload: stress_index, level, affect_state, 4 axes, recovery trend,
  time-in-state, episodes.
- New endpoints: `/api/v1/session/summary`, `/api/v1/insights`.
- Keep the replay/streaming engine and existing endpoints; extend tests.

### Phase 3 — React/Next.js frontend
- Scaffold under `app/` (or `web/`), talking to FastAPI.
- Views: Live (index gauge + 4-axis radar + affect + reasons + quality), Session Summary
  (time-in-state, timeline, recovery), Trends (multi-session), Insights (Tier 4).
- Verify: `npm run build` + API-contract checks. Visual/pixel review is out of scope here.

### Phase 4 — Integration & acceptance
- End-to-end backend + frontend + replay.
- Update `docs/datasets.md`, `docs/model_card_m1.md` (+ a card for the multi-head model), README.
- Extend `scripts/software_acceptance.py`.

## Honesty guardrails (unchanged)
No diagnosis, no panic/burnout prediction. Tier-4 context analysis is descriptive co-occurrence,
never a live cause predictor. Every head reports its real (possibly modest) validation numbers.
