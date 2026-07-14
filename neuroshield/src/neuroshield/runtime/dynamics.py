"""Tier 3: session dynamics derived from the status stream -- no model, just rolling statistics.

Turns a sequence of status records into the summary KPIs the dashboard shows beyond the live
reading: how long was spent in each state, whether stress is currently rising or settling,
distinct stress episodes, and a simple HRV-proxy recovery signal. All of it is arithmetic over
records the engine already produced, so there is nothing to train and nothing to overclaim.
"""

from __future__ import annotations

import numpy as np

TREND_RISING = "rising"
TREND_FALLING = "falling"
TREND_STEADY = "steady"

# States that count as an active stress episode (color-coded elevated/high).
EPISODE_STATES = ("amber", "red")

_TREND_MIN_SLOPE = 1.5  # stress-index points per window to call a trend rising/falling


def _duration(record) -> float:
    if record.window_start_s is None or record.window_end_s is None:
        return 0.0
    return max(0.0, float(record.window_end_s) - float(record.window_start_s))


def time_in_state(records: list) -> dict[str, float]:
    """Total seconds spent in each state across the session (by window duration)."""
    totals: dict[str, float] = {}
    for r in records:
        totals[r.state] = totals.get(r.state, 0.0) + _duration(r)
    return {k: round(v, 1) for k, v in totals.items()}


def recovery_trend(records: list, lookback: int = 5) -> str:
    """Direction of the stress index over the most recent windows that carry an index."""
    indices = [r.stress_index for r in records if r.stress_index is not None]
    if len(indices) < 2:
        return TREND_STEADY
    recent = np.array(indices[-lookback:], dtype=float)
    if len(recent) < 2:
        return TREND_STEADY
    slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
    if slope >= _TREND_MIN_SLOPE:
        return TREND_RISING
    if slope <= -_TREND_MIN_SLOPE:
        return TREND_FALLING
    return TREND_STEADY


def stress_episodes(records: list, min_windows: int = 2) -> list[dict]:
    """Contiguous runs of elevated/high (amber/red) states lasting at least ``min_windows``."""
    episodes = []
    run = []
    for r in records:
        if r.state in EPISODE_STATES:
            run.append(r)
        else:
            if len(run) >= min_windows:
                episodes.append(_episode(run))
            run = []
    if len(run) >= min_windows:
        episodes.append(_episode(run))
    return episodes


def _episode(run: list) -> dict:
    indices = [r.stress_index for r in run if r.stress_index is not None]
    return {
        "start_s": run[0].window_start_s,
        "end_s": run[-1].window_end_s,
        "n_windows": len(run),
        "peak_index": int(max(indices)) if indices else None,
        "peak_state": "red" if any(r.state == "red" for r in run) else "amber",
    }


def hrv_proxy_recovery(records: list, lookback: int = 5) -> float | None:
    """Mean recent pulse-variability (ibi_rmssd_ms) -- higher suggests more parasympathetic recovery."""
    values = [
        r.values.get("ibi_rmssd_ms")
        for r in records
        if r.values and r.values.get("ibi_rmssd_ms") is not None
    ]
    values = [v for v in values if v == v]  # drop NaN
    if not values:
        return None
    return round(float(np.mean(values[-lookback:])), 1)


def session_summary(records: list) -> dict:
    """Combine the Tier-3 dynamics into one JSON-ready summary object."""
    colored = [r for r in records if r.stress_index is not None]
    indices = [r.stress_index for r in colored]
    return {
        "n_windows": len(records),
        "n_scored_windows": len(colored),
        "time_in_state": time_in_state(records),
        "recovery_trend": recovery_trend(records),
        "hrv_proxy_recovery": hrv_proxy_recovery(records),
        "episodes": stress_episodes(records),
        "index_summary": {
            "mean": round(float(np.mean(indices)), 1) if indices else None,
            "max": int(max(indices)) if indices else None,
            "latest": indices[-1] if indices else None,
        },
    }
