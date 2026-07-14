"""Pure display-logic helpers for the dashboard, kept separate from Streamlit rendering.

Streamlit itself can't be meaningfully unit tested at the pixel level, so all decisions --
what label/color a state gets, how quality/values/history get shaped for display -- live here as
plain functions the test suite can exercise directly.
"""

from __future__ import annotations

import pandas as pd

DISCONNECTED = "disconnected"
BACKEND_ERROR = "backend_error"

# (display label, color) for every state the dashboard can show, including the two client-side
# states (disconnected / backend_error) that never come from the backend's own state machine.
STATE_DISPLAY = {
    "waiting": ("Waiting for data", "gray"),
    "calibrating": ("Calibrating baseline", "blue"),
    "green": ("Green - calm", "green"),
    "amber": ("Amber - elevated", "orange"),
    "red": ("Red - high", "red"),
    "motion_paused": ("Motion paused", "gray"),
    "poor_signal": ("Poor signal", "gray"),
    "stale": ("Stale - no recent data", "orange"),
    "error": ("Backend reported an error", "red"),
    DISCONNECTED: ("Disconnected", "red"),
    BACKEND_ERROR: ("Backend error", "red"),
}


def label_and_color(state: str) -> tuple[str, str]:
    return STATE_DISPLAY.get(state, (state, "gray"))


def quality_row(status: dict) -> dict:
    """Extract the quality/coverage metrics row from a status record for display."""
    quality = status.get("quality") or {}
    return {
        "valid_fraction": quality.get("valid_fraction"),
        "ppg_quality": quality.get("ppg_quality"),
        "motion_dynamic_rms": quality.get("motion_dynamic_rms"),
        "motion_dynamic_p95": quality.get("motion_dynamic_p95"),
    }


def latest_values_row(status: dict) -> dict:
    """Extract the raw feature 'values' row (e.g. heart rate, EDA level) for display."""
    return dict(status.get("values") or {})


def history_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten a list of status records into a DataFrame suitable for st.line_chart etc."""
    rows = []
    for r in records:
        row = {
            "window_start_s": r.get("window_start_s"),
            "state": r.get("state"),
            "probability": r.get("probability"),
        }
        row.update(r.get("values") or {})
        rows.append(row)
    df = pd.DataFrame(rows)
    if "window_start_s" in df.columns:
        df = df.sort_values("window_start_s").reset_index(drop=True)
    return df


def is_color_state(state: str) -> bool:
    return state in ("green", "amber", "red")


def is_abstention_state(state: str) -> bool:
    return state in ("motion_paused", "poor_signal")
