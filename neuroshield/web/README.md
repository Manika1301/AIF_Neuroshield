# NeuroShield web dashboard (React / Next.js)

The React/Next.js frontend for NeuroShield (design doc tasks D8/D9). It talks to the FastAPI
backend and provides four views: **Live** (stress index, level, affect, four physiological axes,
reasons), **Summary** (recovery trend, time-in-state, episodes), **Trends** (stress index over the
session), and **Insights** (3-dataset validation scoreboard + nurse context analytics).

## ⚠️ Verification status

This source was written in an environment **without Node.js/npm**, so it has **not** been built or
run here. It is delivered as complete, reviewed source. Before relying on it, run:

```bash
cd web
npm install
npm run typecheck   # tsc --noEmit
npm run build       # next build
```

The **Streamlit dashboard** (`app/dashboard.py`) is the verified UI for the same backend and is
exercised by the automated test suite (`tests/test_dashboard.py`); use it if you need a
known-working dashboard immediately.

## Run (after the backend is up)

```bash
# 1. Start the backend (from the repo root)
uv run python -m neuroshield.api.main         # http://127.0.0.1:8000

# 2. Start the frontend
cd web
npm install
NEXT_PUBLIC_NEUROSHIELD_API_URL=http://127.0.0.1:8000 npm run dev   # http://localhost:3000
```

Then click **Start replay + calibrate** to drive the committed fixture through the pipeline.

## Structure

- `lib/api.ts` — typed backend client; validates `schema_version` / `feature_version` and raises
  `BackendUnreachableError` / `BackendValidationError` (mirrors `app/backend_client.py`).
- `lib/state.ts` — state label/color helpers (mirrors `app/view_state.py`).
- `app/page.tsx` — the four-view dashboard (polls every 3s; shows a disconnected/backend-error
  banner instead of stale data on failure).
- `app/layout.tsx` — root layout.

## Config

Set `NEXT_PUBLIC_NEUROSHIELD_API_URL` to point at the backend (defaults to
`http://127.0.0.1:8000`).
