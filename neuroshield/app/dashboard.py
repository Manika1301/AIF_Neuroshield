"""NeuroShield local dashboard (Streamlit).

Run with: uv run streamlit run app/dashboard.py

All rendering logic here is intentionally thin -- the actual HTTP handling and validation live in
backend_client.py, and display formatting lives in view_state.py, so that logic is unit-testable
even though the rendered page itself is not (no browser automation is available in this
environment; this file was smoke-tested by booting the real server, see T17 notes).
"""

from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from backend_client import (
    DEFAULT_BASE_URL,
    BackendClient,
    BackendUnreachableError,
    BackendValidationError,
)
from view_state import (
    BACKEND_ERROR,
    DISCONNECTED,
    history_to_dataframe,
    is_abstention_state,
    is_color_state,
    label_and_color,
    latest_values_row,
    quality_row,
)

SESSION_WAIT_TIMEOUT_S = 120.0


def _await_session(client: BackendClient, timeout_s: float = SESSION_WAIT_TIMEOUT_S) -> dict:
    """Poll until the backend has streamed every window of the session (or we give up waiting)."""
    deadline = time.monotonic() + timeout_s
    progress = client.session_progress()
    while not progress["complete"] and time.monotonic() < deadline:
        time.sleep(0.15)
        progress = client.session_progress()
    return progress


st.set_page_config(page_title="NeuroShield", page_icon="🧠", layout="wide")

if "client" not in st.session_state:
    base_url = os.environ.get("NEUROSHIELD_API_URL", DEFAULT_BASE_URL)
    st.session_state.client = BackendClient(base_url=base_url)
client: BackendClient = st.session_state.client

st.title("NeuroShield -- Live Status")

# --- Connection state -------------------------------------------------------------------
# The UI must never keep showing an old confident status once the backend is unreachable or
# sends something invalid: every render starts from a fresh poll, and a failure here stops the
# page before any stale status/history from a previous run is displayed.
health = None
system = None
try:
    health = client.health()
    system = client.system()
except BackendUnreachableError as exc:
    label, color = label_and_color(DISCONNECTED)
    st.error(f"**{label}** -- {exc}")
    st.stop()
except BackendValidationError as exc:
    label, color = label_and_color(BACKEND_ERROR)
    st.error(f"**{label}** -- {exc}")
    st.stop()

st.caption(f"Connected to {client.base_url} | model_version={system['model_version']}")

# --- Sidebar: session + calibration controls --------------------------------------------
with st.sidebar:
    st.header("Session controls")

    source_mode = st.selectbox("Source mode", ["replay", "synthetic"])
    replay_path = None
    duration_sec = 600.0
    if source_mode == "replay":
        replay_path = st.text_input("Replay file path", "data/fixtures/calm_motion_stress.ndjson")
    else:
        duration_sec = st.number_input("Duration (seconds)", value=600.0, min_value=1.0)

    session_id = st.text_input("Session ID", "demo-001")
    seed = st.number_input("Seed", value=0, step=1)

    if st.button("Start session", type="primary"):
        try:
            client.start_session(
                source_mode=source_mode,
                session_id=session_id,
                replay_path=replay_path,
                duration_sec=duration_sec,
                seed=seed,
            )
            st.success("Session started.")
        except (BackendUnreachableError, BackendValidationError) as exc:
            st.error(str(exc))

    st.divider()

    quiet_seconds = st.number_input("Quiet calibration seconds", value=150.0, min_value=1.0)
    if st.button("Start calibration"):
        try:
            result = client.start_calibration(quiet_seconds)
            # Calibration now returns as soon as the baseline exists; the backend streams the
            # session's windows in the background. Streamlit has no push channel, so wait for the
            # stream to finish before rendering -- otherwise the first paint shows a half-empty
            # session that only fills in if the user happens to hit Refresh.
            with st.spinner("Processing session windows..."):
                progress = _await_session(client)
            st.success(
                f"Calibrated on {result['n_accepted_windows']} windows "
                f"({result['accepted_seconds']:.0f}s of quiet data). "
                f"Streamed {progress['n_windows']} windows."
            )
        except (BackendUnreachableError, BackendValidationError) as exc:
            st.error(str(exc))

    st.divider()
    history_limit = st.slider("History window (records)", min_value=5, max_value=200, value=40)
    if st.button("Refresh"):
        st.rerun()

# --- Current status ----------------------------------------------------------------------
try:
    status = client.status_latest()
    records = client.history(limit=history_limit)
except BackendUnreachableError as exc:
    label, color = label_and_color(DISCONNECTED)
    st.error(f"**{label}** -- lost connection while fetching status: {exc}")
    st.stop()
except BackendValidationError as exc:
    label, color = label_and_color(BACKEND_ERROR)
    st.error(f"**{label}** -- {exc}")
    st.stop()

label, color = label_and_color(status["state"])
badge_fn = {"green": st.success, "orange": st.warning, "red": st.error}.get(color, st.info)
badge_fn(f"### {label}")

col_conn, col_status, col_quality = st.columns(3)
with col_conn:
    st.metric("Connection", "Connected")
    st.metric("Model loaded", str(health["model_loaded"]))
    st.metric("Baseline calibrated", str(health["baseline_loaded"]))
with col_status:
    st.metric("State", status["state"])
    prob = status.get("probability")
    st.metric("P(stress)", f"{prob:.2f}" if prob is not None else "n/a")
    st.metric("Model version", status.get("model_version") or "n/a")
with col_quality:
    st.write("**Quality**")
    st.json(quality_row(status), expanded=False)

# --- Tier 2: graded index + affect + Tier 1 axes ------------------------------------------
st.subheader("Stress index & physiology")
idx_col, level_col, affect_col = st.columns(3)
with idx_col:
    index = status.get("stress_index")
    st.metric("Stress index (0-100)", index if index is not None else "n/a")
with level_col:
    st.metric("Level", status.get("level") or "n/a")
with affect_col:
    st.metric("Affect state", status.get("affect_state") or "n/a")

axes = status.get("axes") or {}
if axes:
    axis_rows = [
        {"axis": name, "score": a.get("score"), "level": a.get("level")}
        for name, a in axes.items()
    ]
    st.write("**Four physiological axes** (arousal direction, vs. your baseline)")
    st.dataframe(pd.DataFrame(axis_rows), hide_index=True)

# --- Reasons -------------------------------------------------------------------------------
st.subheader("Reasons")
reasons = status.get("reasons") or []
if reasons:
    for reason in reasons:
        st.markdown(f"- {reason}")
elif is_color_state(status["state"]) or is_abstention_state(status["state"]):
    st.caption("No reasons reported for this window.")
else:
    st.caption("No reasons yet -- waiting for the first processed window.")

# --- Latest values ---------------------------------------------------------------------------
st.subheader("Latest values")
values = latest_values_row(status)
if values:
    st.dataframe(pd.DataFrame([values]), hide_index=True)
else:
    st.caption("No feature values yet.")

# --- Short history + chart --------------------------------------------------------------------
st.subheader(f"History (last {len(records)} records)")
if records:
    df = history_to_dataframe(records)
    chart_cols = [c for c in ("probability", "hr_mean_bpm", "eda_level") if c in df.columns]
    if chart_cols and "window_start_s" in df.columns:
        st.line_chart(df.set_index("window_start_s")[chart_cols])
    st.dataframe(df, hide_index=True)
else:
    st.caption("No history yet -- start a session and run calibration.")

# --- Tier 3: session summary ------------------------------------------------------------------
st.subheader("Session summary")
try:
    summary = client.session_summary()
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Recovery trend", summary.get("recovery_trend", "n/a"))
    scol2.metric("Peak index", (summary.get("index_summary") or {}).get("max") or "n/a")
    scol3.metric("HRV-proxy recovery", summary.get("hrv_proxy_recovery") or "n/a")
    st.write("**Time in state (seconds)**")
    st.json(summary.get("time_in_state", {}), expanded=False)
    episodes = summary.get("episodes", [])
    st.caption(f"{len(episodes)} sustained stress episode(s) detected.")
except (BackendUnreachableError, BackendValidationError) as exc:
    st.caption(f"Summary unavailable: {exc}")

# --- Tier 4: research insights ----------------------------------------------------------------
with st.expander("Research insights (descriptive only)"):
    try:
        ins = client.insights()
        st.caption(ins.get("note", ""))
        scoreboard = ins.get("validation_scoreboard")
        if scoreboard:
            st.write("**3-dataset validation scoreboard**")
            st.json(scoreboard.get("per_dataset", {}), expanded=False)
        nurse_md = ins.get("nurse_context_insights_markdown")
        if nurse_md:
            st.markdown(nurse_md)
    except (BackendUnreachableError, BackendValidationError) as exc:
        st.caption(f"Insights unavailable: {exc}")
