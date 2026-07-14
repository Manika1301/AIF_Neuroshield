# Product scope: software-first MVP

This document states what release 1 (the software-first MVP) must do, and what is explicitly
postponed until the software replay gate passes. It exists so anyone opening the repository can
tell, without reading code, what is being built now versus later.

## Required release-1 behavior

The MVP is a laptop application that runs entirely on public datasets and a generated/replayed
sensor stream. To be considered done, all of the following must exist and work:

- **Dataset ingestion** — WESAD loads correctly. At least one additional external dataset loader
  exists, or a documented reason for skipping it is recorded.
- **Feature pipeline** — one versioned feature extractor (`features-v1`) converts signal windows
  into heart, EDA, temperature, and motion features with a fixed column order.
- **Model** — an M1 rest-vs-stress-proxy classifier trained on WESAD with subject-wise
  (leave-one-subject-out) evaluation, compared against a dummy baseline, saved as a versioned
  artifact with a manifest.
- **Synthetic/replay stream source** — a software-only stream generator that emits the exact raw
  event schema future hardware will use, plus a replay player that reads recorded/generated
  fixtures in timestamp order.
- **Personalization** — a baseline profile computed from a quiet segment, used to z-score live
  features per user/session.
- **Live runtime** — the replay/synthetic stream flows through feature extraction, baseline
  z-scoring, a motion/signal-quality gate, the model, a status state machine, and plain-language
  explanations.
- **Dashboard** — a local dashboard showing connection state, signal quality, the
  green/amber/red status, motion-paused/poor-signal abstention, ranked reasons, and short
  history.
- **Tests and evidence** — dataset smoke tests, feature tests, model metrics, backend tests, a
  dashboard build, and a recorded replay demo.

## What is deliberately postponed

The following are out of scope until the software replay gate (see
`docs/software_acceptance.md`) passes:

- Hardware wiring and sensor integration.
- Firmware development.
- LiPo battery / power system design.
- The Bluetooth phone companion app.
- On-device TinyML / embedded inference.
- A permanent physical enclosure.
- Any clinical claims or clinical validation.
- Panic-attack prediction.
- Real burnout prediction (as opposed to descriptive trend display).
- Volunteer stress experiments involving real users and real hardware.

## Why this order

Building the model, backend, dashboard, and contracts against a software-only stream first means
that when hardware is connected later, it only has to replace the stream *source*. Everything
downstream of the raw event schema — features, model, baseline, runtime states, API, and UI — is
already built and tested and should not need to change.
