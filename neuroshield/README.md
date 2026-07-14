# NeuroShield

**MVP statement: software replay first, hardware source later.**

NeuroShield is being built as a software-first research prototype. The first milestone is a
complete laptop application that works end to end using public physiological datasets and a
generated/replayed sensor stream — no wearable hardware required. Only after that replay demo is
stable and passes the software acceptance gate (see `docs/software_acceptance.md`) does hardware
integration begin.

When hardware does arrive, it is designed to replace the stream source only. The feature
pipeline, model, backend, dashboard, logging, explanations, and tests are all built and tested
against the replay/synthetic source first, so they should not need to change when a real device
is connected — see `docs/hardware_handoff.md` (added once the hardware contract task runs).

## Why software first

- Lets the model, backend, and dashboard be built, tested, and demoed without waiting on
  hardware, firmware, or a physical build.
- Forces the raw event schema and contracts to be defined up front (`docs/contracts.md`), so a
  synthetic generator and future firmware speak the exact same protocol.
- Produces a reproducible, versioned pipeline (datasets -> features -> model -> runtime ->
  dashboard) with tests and saved artifacts before any device-specific complexity is introduced.

## What this repository contains right now

- `docs/` — contracts, dataset notes, model card, and the software acceptance definition.
- `src/neuroshield/` — Python package: dataset loaders, feature extraction, models, live
  runtime, and the FastAPI backend.
- `app/` — the local dashboard.
- `tests/` — automated tests for the pipeline.
- `data/` — dataset storage. `raw/`, `external/`, and `interim/` are not committed; `fixtures/`
  holds small synthetic replay files that are committed.
- `artifacts/` — generated metrics, plots, trained model files, and demo evidence.

## What is postponed

Hardware wiring, firmware, LiPo power, the Bluetooth phone app, on-device TinyML, a permanent
enclosure, clinical claims, panic prediction, real burnout prediction, and volunteer stress
experiments are all postponed until the software replay gate passes. See
`docs/product_scope.md` for the full breakdown.

## Honesty boundary

NeuroShield is not a clinical device. See `docs/no_clinical_claims.md`.

## Getting started

Full dataset download instructions live in `docs/datasets.md`; the original task plan is in
`tasks.tex` one level up from this directory. To reproduce the software-only replay demo from a
clean checkout:

```bash
# 1. Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/

# 2. Sync the environment and confirm it's healthy
uv sync
uv run python -m neuroshield.smoke

# 3. Run the full automated test suite
uv run pytest tests/ -q

# 4. Run the automated software acceptance gate (T19) -- generates
#    artifacts/demo/software_acceptance.json
uv run python scripts/software_acceptance.py

# 5. Start the backend (binds to 127.0.0.1:8000)
uv run python -m neuroshield.api.main

# 6. In another terminal, start a replay session against the committed fixture and calibrate
curl -X POST http://127.0.0.1:8000/api/v1/session/start \
  -H "Content-Type: application/json" \
  -d '{"source_mode":"replay","replay_path":"data/fixtures/calm_motion_stress.ndjson","session_id":"demo-001"}'
curl -X POST http://127.0.0.1:8000/api/v1/calibration/start \
  -H "Content-Type: application/json" -d '{"quiet_seconds":150}'

# 7. In a third terminal, start the dashboard and open the printed local URL
uv run streamlit run app/dashboard.py
```

The backend now serves the **multi-head model** (`m3_multihead_personalized_v1`): a 0-100 stress
index + calm/elevated/high level (Head A) and a baseline/stress/amusement/meditation affect state
(Head B), plus the four physiological axes and Tier-3 session dynamics. Steps 4-8 require its
artifact at `artifacts/models/m3_multihead_personalized_v1.joblib` (gitignored). If you don't have it,
download WESAD + Stress-Predict per `docs/datasets.md` and run:

```bash
uv run python scripts/train_multihead.py          # trains + freezes the multi-head artifact
uv run python scripts/build_scoreboard.py          # 3-dataset validation scoreboard (needs Nurse data for the held-out row)
uv run python -m neuroshield.models.nurse_insights # Tier-4 nurse context analytics
```

The historical single-head M1 model (`m1_wesad_features_v1`, `uv run python -m
neuroshield.models.artifact`) is retained for comparison and is never overwritten.

### Frontends

- **Streamlit** (`app/dashboard.py`) -- the verified UI, exercised by the test suite. Shows the
  index, level, affect, four axes, session summary, and research insights.
- **React / Next.js** (`web/`) -- the richer frontend from the design doc. Delivered as reviewed
  source; see `web/README.md`. It was **not** built in this environment (no Node/npm), so run
  `npm install && npm run build` before relying on it.

See `docs/design_doc.tex` for the full end-to-end design and `docs/software_acceptance.md` for the
acceptance procedure.
