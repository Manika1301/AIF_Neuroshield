# NeuroShield — Project Context

This file exists so a fresh Claude Code session can pick up this project with zero prior context. Read this first before touching any code or docs.

## What NeuroShield is

A wearable-tech research project: a biosensor glove + companion app that aims to detect physiological/mental-health crises **before** they happen — panic attacks, heat exhaustion, cognitive overload, and burnout. Built by Manika Jhaveri (MIT Summer Research Program, June 2026).

**Read this framing gap before doing anything else** — it's the single most important thing to understand:

- The **pitch/marketing materials** (deck, docx pitch docs, literature review) describe an ambitious end-state: an LSTM model predicting panic attacks 60 minutes out, a Kalman-filter core-body-temperature estimator, burnout prediction, branded as four features — **Panic Shield, HeatGuard, FocusGuard, BurnoutWatch**.
- The **actual engineering repo** (`neuroshield/`) is deliberately more conservative. `neuroshield/docs/no_clinical_claims.md` explicitly forbids claiming the system "predicts or detects a panic attack," "diagnoses heat illness," or "detects/predicts burnout." The real, built models are logistic regression / HistGradientBoostingClassifier trained on **proxy labels** (rest vs. stress-proxy from public datasets) — there is no LSTM, no Kalman filter, and no clinical validation anywhere in the code.
- Treat the pitch deck as the **aspirational vision** and the `neuroshield/` repo as the **honest, current MVP substrate** for it. Don't let pitch-deck claims leak into engineering docs or code comments as if they were already true.

## Problem statement (as argued across the docs)

- ~970M people worldwide live with a diagnosable mental disorder; 40% of employees report significant daily stress.
- Depression/anxiety cost the global economy ~$1T/year and 1.2B lost workdays; projected to reach $16T by 2030.
- Core argument: existing wearables (Fitbit, Apple Watch, Oura, WHOOP) only *report* physiological state after the fact against population norms, with an opaque score — they don't predict, personalize, or explain. The Empatica E4 (~$1,600, research-grade) comes closest but is lab-only.

## Product concept

**The glove**: fingertip/palm-mounted (deliberately not wrist — argued as a better signal site), five target sensors:
- MAX30102 — PPG / pulse (heart rate, HRV)
- ADS1115 + Ag/AgCl electrodes — EDA (skin conductance)
- MLX90614 — skin temperature
- BNO055 — IMU / motion
- AD8232 — single-lead ECG (optional)

No screen on the glove; streams to a companion app over Bluetooth/USB/Wi-Fi.

**Four branded crisis features** (pitch-deck terminology):
| Feature | Detects | Warning window | Claimed method |
|---|---|---|---|
| Panic Shield | Panic/anxiety spikes | Up to 60 min ahead | "LSTM" on autonomic cascade (HRV drop, EDA rise, temp drop) |
| HeatGuard | Heat exhaustion | 15–30 min ahead | Kalman-filter core-body-temp estimate (Eggenberger et al. 2018 Min-Input Model, SEE 0.29°C) |
| FocusGuard | Cognitive overload | Real-time (30s window) | Per-user random forest, Green/Amber/Red Focus Score |
| BurnoutWatch | Long-term burnout | 2–4 weeks ahead | Weekly trend vs. user's own 4-week rolling average (HRV/RMSSD, deep sleep %, resting HR, steps, HRV recovery) |

**Companion app** (per pitch docs): home screen (live readings), alert screen (guided breathing/cooldown/break interventions), dashboard/trend view, baseline-calibration screen, event log.

## Technical architecture

### Hardware — early bring-up stage, this is the real gap
- Dev MCU: ESP32-S3 (Wi-Fi/USB, shared I2C bus, GPIO8/9 = SDA/SCL). Production target: Seeed XIAO nRF52840 (BLE, lower power) — **not yet ordered**.
- **Only firmware that actually exists**: [`/Users/manikajhaveri/Downloads/neuroshield_max30102/neuroshield_max30102.ino`](/Users/manikajhaveri/Downloads/neuroshield_max30102/neuroshield_max30102.ino) — a 2-sensor bench-test sketch (MAX30102 + MLX90614 only, via SparkFun `MAX30105`/`heartRate.h`). Computes BPM via peak detection over an 8-sample rolling average, rejects BPM outside 40–180, flags "No finger detected" if IR < 50000, prints heart rate/IR/temp over Serial (115200 baud, 1×/sec).
- This sketch prints **plain human-readable Serial text**, not the NDJSON event contract (see below) the software side expects. No `serial_source.py` adapter exists yet to bridge them.
- Full 5-sensor glove assembly is documented (parts list, breadboard→perfboard→glove wiring, power via TP4056+LiPo) in the Downloads docx files (see inventory below), but nothing beyond the 2-sensor bench sketch has been built.
- Per `neuroshield/docs/hardware_handoff.md`: "hardware acceptance" only begins once a real device emits valid `neuroshield.hw.v1` events — by that gate, hardware integration has not started.

### Software / ML pipeline — mature, tested, this is the substantive part
Repo: `neuroshield/` (inside this directory). Python ≥3.10, key deps: `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `neurokit2`, `numpy/pandas/scipy`, `joblib`, `pyserial`, `streamlit`, `pyarrow`, `cvxopt` (cvxEDA tonic/phasic decomposition).

- **Raw event contract** — `neuroshield/docs/contracts.md`, schema `neuroshield.hw.v1`: NDJSON, one event/line, types `ppg`(64Hz)/`eda`(4Hz)/`temp`(4Hz)/`imu`(32Hz)/`health`(~1Hz). This is the seam meant to let real hardware swap in for the synthetic/replay source with zero downstream changes.
- **Data loaders** (`src/neuroshield/data/`): `wesad_loader.py`, `stress_predict_loader.py`, `nurse_stress_loader.py`, `bundle.py` (canonical `SignalBundle`).
- **Features** (`src/neuroshield/features/`): `extract.py` (features-v1, 13 cols: hr_mean_bpm, ibi_sd_ms, ibi_rmssd_ms, ppg_quality, eda_level, eda_slope, eda_response_count, eda_response_mean_amp, temp_mean_c, temp_slope_c_per_min, motion_dynamic_rms, motion_dynamic_p95, valid_fraction), `harmonize.py`, `labels.py`, `personalize.py` (per-user z-scoring → features-v2, 36 cols).
- **Models** (`src/neuroshield/models/`): `train_m1.py` (LOSO logistic regression), `artifact.py` (versioned model + manifest + checksum), `multihead.py` (M3: Head A graded stress + Head B affect), `scoreboard.py` (3-dataset validation), `nurse_insights.py`, `external_validation.py`.
- **Runtime** (`src/neuroshield/runtime/`): `synthetic_source.py`, `replay_source.py`, `events_to_bundle.py`, `baseline.py` (personal z-score), `quality_gate.py` (motion/signal abstention), `status.py` (state machine: waiting/calibrating/green/amber/red/motion_paused/poor_signal/stale/error), `explain.py` (feature-grounded reasons), `axes.py` (4-axis decomposition), `dynamics.py`.
- **API** (`src/neuroshield/api/`): FastAPI, `127.0.0.1:8000`. Endpoints: `GET /api/v1/health`, `/system`, `POST /session/start`, `/calibration/start`, `GET /status/latest`, `/history`, `/session/summary`, `/insights`, `WS /ws/v1/live`.
- **Scripts** (`scripts/`): `build_scoreboard.py`, `cache_wesad_features.py`, `software_acceptance.py` (T19 acceptance gate), `train_multihead.py`.
- **Tests**: 27 files, README claims 253 passing at time of writing.

**Model metrics actually achieved:**
- M1 (logistic regression, features-v1, WESAD-only): LOSO balanced acc. **0.831**, macro-F1 0.822 (dummy baseline 0.500). Cross-dataset transfer to Stress-Predict: balanced acc. 0.541 (modest — different stress inducer).
- M3/multihead (HistGradientBoostingClassifier, features-v2, 36 cols): Head A (graded stress) LOSO balanced acc. **0.919** vs. dummy 0.500; Head B (4-class affect) 0.616 vs. dummy 0.250.
- Progression: 0.831 (M1, WESAD-only) → 0.617 (naive pooling with Stress-Predict — **hurt accuracy, abandoned**) → 0.831 (WESAD-only, confirmed) → 0.880 (+features-v2) → 0.919 (+personalization). The naive-pooling-hurts-accuracy finding is a concrete decision worth remembering.
- Nurse Stress dataset (15 nurses, ~620 sessions, 225 labelled events) held out entirely from training — validation + descriptive insights only (e.g., "treating a COVID patient" stress co-occurrence lift 2.06).

### Frontend
- `app/dashboard.py` — **Streamlit, the only verified/working UI**, covered by `tests/test_dashboard.py`.
- `web/` — Next.js/React (`app/page.tsx`, `lib/api.ts` typed client w/ schema-version validation, `lib/state.ts`) — **reviewed source only, never built/run** (no Node available in the build environment). Flagged as a verification gap in `design_doc.tex`.
- `/Users/manikajhaveri/Downloads/neuroshield app prototype.html` + `neuroshield-app-prototype-screenshot.png` — a separate rough phone-app HTML mockup, not integrated with the above.

### Datasets on disk (`neuroshield/data/`)
- WESAD — full raw data, subjects S2–S17 (S1/S12 skipped per official release).
- Stress-Predict — full clone, 35 subjects (S01–S35), raw + processed + questionnaires.
- Nurse Stress (Dryad) — `Stress_dataset.zip` (~1.16GB, read in-memory, never extracted) + `SurveyResults.xlsx`.
- `data/fixtures/calm_motion_stress.ndjson` — committed synthetic replay fixture (600s, seed 7) driving the whole software-acceptance gate.
- PPG-DaLiA, SWELL-KW, Fitbit — **not downloaded**; skip decisions documented in `docs/datasets.md`, deferred to optional tasks O1–O3.

## Research basis

- **`Neuroshield research paper draft.pdf`** (Downloads root) — 5-page ACM-format paper, properly cited (17 real refs: Schmidt/WESAD 2018, Taylor et al. personalized multitask learning, Lundberg/SHAP, van Dooren electrode-site study, Herborn skin-temp/stress, Prajod cross-dataset study, etc.). Abstract explicitly states results are **"the subject of ongoing work"** — all result tables say TBD. Describes a 5-stage pipeline (acquire → clean/reject motion → featurize → normalize+infer → explain) and model family M1–M5. States no public dataset has labelled panic/heat-exhaustion/burnout episodes — everything is proxy-label training. Note: author affiliation/email in the draft look like placeholder text (`example.com` domain) — probably not submission-ready.
- **`NeuroShield_Literature_Review.docx`** (Downloads root) — 19+ papers organized by the four pitch features, leans more into LSTM/panic-prediction claims than the paper draft does.
- **`NeuroShield datasets.pdf`** (Downloads root) — dataset-scouting doc listing candidates (TILES-2018/2019, ADARP, a 36-subject PhysioNet/Zenodo set, a 29-subject 2024 Data-in-Brief set) that don't appear anywhere else in the project — likely an earlier/parallel scouting pass, superseded by `neuroshield/docs/datasets.md`.

## Current status

- **Git state** (`ai_neuroshield/AIF_Neuroshield`, branch `main`): only 2 commits ("Initial commit", "Uploaded the research draft"). Everything substantive — `design_doc.tex`, `build_plan_sprint_format.tex`, `tasks.tex`, and the entire `neuroshield/` folder — is **untracked/uncommitted**. Worth committing in logical chunks before it's at risk of being lost; ask the user before doing so.
- **Software-first MVP (T1–T20 in `tasks.tex`)**: marked fully DONE in `design_doc.tex`.
- **Phase-2 redesign (D0–D10 in `design_doc.tex`)**: also marked fully DONE — multihead model (M3), 4-axis decomposition, 3-dataset scoreboard, nurse Tier-4 insights, enriched API, React scaffold (unbuilt).
- **Hardware**: not integrated — only the 2-sensor bench sketch exists (see above).
- **Overall stage**: software/ML pipeline is mature and well-tested; hardware/firmware is at early bring-up; pitch/research materials describe the aspirational end-state, ahead of what's implemented.

## Known inconsistencies / things to ask the user before acting on

1. **Pitch vs. engineering claims** (see top of this file) — don't let pitch-deck claims (LSTM, Kalman filter, clinical prediction) leak into code/docs as fact.
2. **Feature-version drift**: `design_doc.tex` (2026-07-11) still describes features-v1 (13 cols) as current, but `docs/model_card_m3_multihead.md` (also current) describes the production model running on features-v2 (36 cols). Doc-sync issue, not a functional bug.
3. `tasks.tex` header references `manika/initial_research.tex`, a path that doesn't exist anywhere in the project — likely stale from an earlier layout.
4. Two near-duplicate hardware docs (`hardware neuroshield.docx` and `neuroshield hardware assembling.docx`) contain the same 23-step assembly procedure almost verbatim — confirm which is canonical before editing either.
5. `NeuroShield.docx` (Downloads root) ends with a bare email (`trishulchowdhury.23@gmail.com`), no context — possibly a collaborator; don't assume, ask if it matters.
6. A pitch deck (`NeuroShield.pptx`) was being visually redesigned in a separate Claude session — check with the user whether that's finished/wanted before touching it again.

## File inventory

### Repo (`/Users/manikajhaveri/Downloads/ai_neuroshield/AIF_Neuroshield/`)
- `design_doc.tex` — full end-to-end architecture/design doc, most current source of truth (dated 2026-07-11).
- `build_plan_sprint_format.tex` — 10-day hands-on build sprint plan (hardware + ML tracks).
- `tasks.tex` — software-first MVP task plan T1–T20 + optional O1–O3, with Definition of Done per task.
- `AIF_Manika.pdf` — (not read in detail; check if relevant when needed).
- `neuroshield/README.md` — repo overview + reproduction steps + current model version pointer.
- `neuroshield/pyproject.toml` — dependency manifest.
- `neuroshield/docs/contracts.md` — raw event schema `neuroshield.hw.v1`.
- `neuroshield/docs/hardware_handoff.md` — firmware builder's contract; defines when hardware acceptance begins.
- `neuroshield/docs/software_acceptance.md` — T19 acceptance gate reproduction guide.
- `neuroshield/docs/model_card_m1.md` — M1 model card.
- `neuroshield/docs/model_card_m3_multihead.md` — M3 model card.
- `neuroshield/docs/product_scope.md` — release-1 required behavior vs. postponed items.
- `neuroshield/docs/datasets.md` — per-dataset status (used/skipped) with justification.
- `neuroshield/docs/build_roadmap.md` — Phase 1–4 roadmap, records decisions locked with the user on 2026-07-11.
- `neuroshield/docs/no_clinical_claims.md` — honesty-boundary doc, explicit forbidden-claims list.
- `neuroshield/src/neuroshield/{features,runtime,models,api,data}/`, `neuroshield/app/`, `neuroshield/web/`, `neuroshield/scripts/`, `neuroshield/tests/`, `neuroshield/data/` — see Technical Architecture above.

### Firmware
- `/Users/manikajhaveri/Downloads/neuroshield_max30102/neuroshield_max30102.ino` — 2-sensor bring-up sketch (see Hardware above).

### Downloads root — standalone docs
- `NeuroShield outline .docx` — condensed research-paper outline (problem/solution/architecture/methodology/evaluation/limitations/contributions).
- `NeuroShield problem addressed.docx` — short problem statement, stats-only.
- `NeuroShield.docx` — fullest pitch document: problem, solution, four features in detail, target users, app description. Ends with a stray email (see inconsistencies above).
- `NeuroShield.pdf` — PDF export of the pitch-deck content (matches `NeuroShield.docx` narrative, reformatted).
- `NeuroShield outline .pdf` — PDF export of the outline docx.
- `NeuroShield_Literature_Review.docx` — 19+ paper literature review, organized by the four pitch features.
- `Neuroshield research paper draft.pdf` — 5-page ACM-format paper, real citations, results marked TBD.
- `NeuroShield datasets.pdf` — dataset-scouting doc (see Research Basis above).
- `assemble steps neuroshield.docx` — shopping list + 23-step breadboard→perfboard→glove assembly guide (5-sensor version incl. AD8232 ECG).
- `hardware neuroshield.docx` — parts list + same 23-step assembly guide (near-duplicate of above).
- `neuroshield hardware assembling.docx` — most detailed hardware doc: I2C address table, phased build/test order.
- `neuroshield brendan pitch.docx` — short spoken pitch script, casual tone, "nothing else does this" positioning vs. Apple Watch/Fitbit/Oura/WHOOP/Empatica.
- `neuroshield app prototype.html` + `neuroshield-app-prototype-screenshot.png` — rough phone-app HTML mockup, standalone from `web/`.
- `NeuroShield.pptx` — pitch deck (being visually redesigned in a separate session as of last check).
- `~$rdware neuroshield.docx` — Word temp/lock file, not real content, safe to ignore.

## Suggested first steps for a new session

1. Ask the user what they actually want to work on next (hardware firmware bring-up, ML pipeline improvements, frontend, pitch materials, or research paper) — this project has many active fronts.
2. If touching the `neuroshield/` repo, run its test suite first (`cd neuroshield && pytest`) to confirm the 253-passing baseline still holds before changing anything.
3. If touching git, note everything is currently uncommitted — check with the user before committing/staging broadly, and review `.gitignore` (data dirs, `.venv` etc. should already be excluded).
4. Don't add or reinforce clinical/predictive claims that `no_clinical_claims.md` forbids, even if asked to "match the pitch deck" — flag the tension to the user instead.
