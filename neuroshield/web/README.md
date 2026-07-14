# NeuroShield — web dashboard

React / Next.js 14 (App Router) frontend for the NeuroShield backend.

## Verification status — read this first

**This code has never been compiled.** It was written and reviewed by hand in an environment with no
Node, no npm, and no way to install them. Every other part of NeuroShield (backend, models, Streamlit
dashboard) is verified by a passing test suite and by being run end-to-end; this app is not.

Expect a first `npm run build` to surface type errors. That is normal for 700+ lines of
never-compiled TypeScript and does not mean the design is wrong — report the errors and they can be
fixed quickly. **Do not demo from this until it builds.**

The Streamlit dashboard (`app/dashboard.py`, run from the repo root) is the verified UI and shows the
same data.

## Run it

The backend must be running first:

```bash
# from the repo root
uv run uvicorn neuroshield.api.main:app --port 8000
```

Then:

```bash
cd web
cp .env.example .env.local     # optional; defaults to http://127.0.0.1:8000
npm install
npm run typecheck              # do this first -- fastest way to find breakage
npm run dev                    # http://localhost:3000
```

The backend allows CORS from `localhost:3000` and `127.0.0.1:3000` by default. Serving the frontend
from any other origin requires setting `NEUROSHIELD_CORS_ORIGINS` on the backend, or the browser will
block every request.

## How it gets data

Status is **pushed, not polled**. On mount, `lib/ws.ts` opens a WebSocket to `/ws/v1/live` and
receives one message per 60-second window as the backend processes it, then a `session_complete`
message. It reconnects automatically and replays the backlog on reconnect, so refreshing the browser
mid-session loses nothing.

REST (`lib/api.ts`) is used only for what isn't per-window: system info, the session summary, and the
offline research artifacts.

## Views

| Tab | Shows |
|---|---|
| **Live** | Status badge, the 0–100 stress index (hero KPI), level, affect state + confidence, the four physiological axes as signed bars, plain-language reasons, latest values, signal quality |
| **Trends** | Stress index / heart rate / skin conductance over time (inline SVG — no charting dependency) |
| **Session summary** | Recovery trend, peak/mean index, HRV proxy, stress episodes, time in each state |
| **Research insights** | Cross-dataset validation scoreboard and nurse-shift context |

## Two things the UI is deliberate about

**Abstention is shown, not hidden.** When the model declines to score a window (hand motion, poor
signal), the UI says so explicitly rather than showing a stale or invented number. In the Trends
chart those windows are *gaps in the line* — interpolating across them would draw data the model
explicitly refused to produce.

**Superseded numbers are labelled.** The validation scoreboard on disk predates the shipped model.
The backend flags it, and the Insights tab renders that warning above the table rather than quietly
presenting old numbers as current.

See `docs/prediction_spec.md` for what every field means, and `docs/no_clinical_claims.md` for what
this product may not claim.
