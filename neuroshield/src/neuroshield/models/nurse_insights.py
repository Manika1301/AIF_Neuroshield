"""D5: descriptive Tier-4 analytics -- which contexts co-occur with high-stress nurse reports.

This is deliberately *descriptive*, never predictive. The Nurse Stress survey attaches context
flags (patient crisis, workload, COVID patient, ...) to each self-reported stress event. This
module tabulates, among high-stress reports vs. low-stress reports, how often each context was
cited -- answering "when nurses reported high stress, what was going on?" It does NOT predict a
cause from physiology (that would overclaim; see docs/no_clinical_claims.md).

The core (``context_cooccurrence``) is pure and works on the survey table alone. The optional
real-data path in ``__main__`` additionally annotates each context with the mean measured stress
index over that context's event windows (from the frozen multi-head model), when available.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from neuroshield.data.nurse_stress_loader import CONTEXT_COLUMNS

DEFAULT_INSIGHTS_MD = Path("artifacts/metrics/nurse_context_insights.md")

HIGH_STRESS_LEVEL = 2
LOW_STRESS_LEVEL = 0

_TRUTHY = {"1", "1.0", "yes", "y", "true", "x"}

DESCRIPTIVE_DISCLAIMER = (
    "Descriptive only. This table reports how often each context was *cited by nurses* alongside "
    "high vs. low self-reported stress. It is not a physiological cause detector and must never be "
    "presented as one (see docs/no_clinical_claims.md)."
)


def _is_flagged(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return str(value).strip().lower() in _TRUTHY


def context_cooccurrence(events: pd.DataFrame, context_columns: list[str] = None) -> pd.DataFrame:
    """Per-context co-occurrence with high- vs low-stress reports, ranked by high-stress rate.

    ``events`` must have a ``stress_level`` column (0/1/2/'na') and the context flag columns.
    Returns a DataFrame with one row per context: counts and rates among high (level 2) and low
    (level 0) stress events, plus a ``lift`` (how many times more common in high vs low).
    """
    context_columns = context_columns or [c for c in CONTEXT_COLUMNS if c in events.columns]
    high = events[events["stress_level"] == HIGH_STRESS_LEVEL]
    low = events[events["stress_level"] == LOW_STRESS_LEVEL]
    n_high = max(len(high), 1)
    n_low = max(len(low), 1)

    rows = []
    for col in context_columns:
        high_flagged = int(high[col].map(_is_flagged).sum()) if col in high else 0
        low_flagged = int(low[col].map(_is_flagged).sum()) if col in low else 0
        rate_high = high_flagged / n_high
        rate_low = low_flagged / n_low
        lift = (rate_high / rate_low) if rate_low > 0 else float("inf") if rate_high > 0 else 0.0
        rows.append(
            {
                "context": col,
                "n_high_stress": high_flagged,
                "rate_high_stress": round(rate_high, 3),
                "n_low_stress": low_flagged,
                "rate_low_stress": round(rate_low, 3),
                "lift_high_vs_low": (round(lift, 2) if lift != float("inf") else "inf"),
            }
        )
    table = pd.DataFrame(rows).sort_values("rate_high_stress", ascending=False).reset_index(drop=True)
    return table


def render_insights_markdown(table: pd.DataFrame, n_high: int, n_low: int, extra_note: str = "") -> str:
    lines = [
        "# Nurse Stress: context co-occurrence with self-reported stress (Tier 4)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"High-stress events (level 2): {n_high} | Low-stress events (level 0): {n_low}",
        "",
        "| Context | High-stress n | High rate | Low-stress n | Low rate | Lift (high/low) |",
        "|---|---|---|---|---|---|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.context} | {row.n_high_stress} | {row.rate_high_stress} | "
            f"{row.n_low_stress} | {row.rate_low_stress} | {row.lift_high_vs_low} |"
        )
    lines += ["", "## Note", "", DESCRIPTIVE_DISCLAIMER]
    if extra_note:
        lines += ["", extra_note]
    return "\n".join(lines)


def save_insights(table: pd.DataFrame, n_high: int, n_low: int, path: Path = DEFAULT_INSIGHTS_MD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_insights_markdown(table, n_high, n_low))


if __name__ == "__main__":
    from neuroshield.data.nurse_stress_loader import load_survey_events

    try:
        events = load_survey_events()
    except FileNotFoundError as exc:
        raise SystemExit(f"Nurse Stress data not available: {exc}") from exc

    table = context_cooccurrence(events)
    n_high = int((events["stress_level"] == HIGH_STRESS_LEVEL).sum())
    n_low = int((events["stress_level"] == LOW_STRESS_LEVEL).sum())
    save_insights(table, n_high, n_low)
    print(render_insights_markdown(table, n_high, n_low))
