# NeuroShield — dashboard

Next.js 14 (App Router) + Tailwind + [shadcn/ui](https://ui.shadcn.com). The only frontend; the
Streamlit prototype has been removed.

## Run it

The backend must be running first:

```bash
# from the repo root
uv run uvicorn neuroshield.api.main:app --port 8000
```

Then:

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

Open it and press **Start session** — the dashboard drives the whole flow (start → calibrate →
stream). No curl required.

The backend allows CORS from `localhost:3000` and `127.0.0.1:3000`. Serving from any other origin
means setting `NEUROSHIELD_CORS_ORIGINS` on the backend, or the browser blocks every request.

## The design goal: it has to be understandable

A 0–100 "stress" number, on its own, is an invitation to misread. So:

- **The number never appears alone.** It always carries a plain-English verdict ("High — arousal is
  well above your resting level"), what it's relative to (*your* baseline, not a population), and
  the reasons behind it.
- **Jargon is translated once**, in `lib/state.ts`. The API says `electrodermal`, `poor_signal`,
  `hrv_proxy_recovery`; the UI says "sweat response", "paused — weak signal", "recovery signal".
  That vocabulary stays correct in the API and never leaks to the screen.
- **Refusals are shown as refusals.** When the model declines to score a window (hand motion, poor
  contact) the UI says so and explains why, instead of showing a stale number. In the chart those
  windows are **gaps in the line** — interpolating across them would draw data the model explicitly
  refused to produce.
- **The error bar is in the product**, not buried in a docs folder. The "How it works" tab states
  the accuracy (0.919 ±0.036) *and* that per-person accuracy ranged from 0.50 to 1.00 — i.e. this
  model is good on average and can be useless for an individual. A health dashboard that hides that
  is lying by omission.
- **Direction is stated, not assumed.** Cooler skin and *falling* heart-rate variability both mean
  *more* arousal. The axis bars are anchored at the centre and phrased "toward arousal" / "toward
  calm", because "above your baseline" would be actively wrong for the thermal axis.

## How data arrives

Status is **pushed, not polled**. `lib/ws.ts` opens a WebSocket to `/ws/v1/live` and receives one
message per 60-second window as the backend computes it, then `session_complete`. It reconnects
automatically and replays the backlog, so a refresh mid-session loses nothing.

REST (`lib/api.ts`) is used only for what isn't per-window: system info, session summary, and the
offline research artifacts.

## Layout

| File | Role |
|---|---|
| `app/page.tsx` | Shell, tabs, session wiring |
| `components/StatusHero.tsx` | The verdict + index + why |
| `components/AxisBars.tsx` | The four measured systems |
| `components/SessionSetup.tsx` | One-click start (advanced knobs collapsed) |
| `components/AboutModel.tsx` | Accuracy, limits, inputs/outputs |
| `components/TimeSeries.tsx` | Dependency-free SVG chart with real gaps |
| `lib/state.ts` | **The plain-language layer** |

## Checks

```bash
npm run typecheck
npm run lint
npm run build
```

See `docs/prediction_spec.md` for what every field means, and `docs/no_clinical_claims.md` for what
this product may not claim.
