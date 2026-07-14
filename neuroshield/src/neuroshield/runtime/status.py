"""Turn M1's P(stress) into a stable product status, with hysteresis and abstention handling.

Abstention (``motion_paused``, ``poor_signal``) and connectivity states (``waiting``,
``calibrating``, ``stale``, ``error``) always override the color decision immediately -- safety
and honesty come before smoothness. Only the green/amber/red decision itself is debounced: a
single anomalous window is not enough to flip an already-established color, so a momentary spike
doesn't make the status flicker.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from neuroshield.models.artifact import DEFAULT_THRESHOLD_POLICY
from neuroshield.runtime.quality_gate import MOTION_PAUSED, POOR_SIGNAL, AbstentionResult

WAITING = "waiting"
CALIBRATING = "calibrating"
GREEN = "green"
AMBER = "amber"
RED = "red"
STALE = "stale"
ERROR = "error"
COLOR_STATES = {GREEN, AMBER, RED}
ALL_STATES = {WAITING, CALIBRATING, GREEN, AMBER, RED, MOTION_PAUSED, POOR_SIGNAL, STALE, ERROR}

DEFAULT_HYSTERESIS_WINDOWS = 2
DEFAULT_STALE_GAP_S = 120.0


@dataclass
class StatusRecord:
    timestamp: str
    state: str
    probability: float | None
    model_version: str | None
    feature_version: str | None
    quality: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    window_start_s: float | None = None
    window_end_s: float | None = None
    # Enriched (multi-head) fields -- populated by the engine when the multi-head model is loaded.
    stress_index: int | None = None  # 0-100 calibrated graded stress
    level: str | None = None  # calm / elevated / high
    affect_state: str | None = None  # baseline / stress / amusement / meditation
    axes: dict = field(default_factory=dict)  # cardiac / electrodermal / thermal / movement

    def to_dict(self) -> dict:
        return _sanitize_for_json(asdict(self))


def _sanitize_for_json(value):
    """Replace NaN/Infinity with None -- standard JSON (and Starlette's encoder) rejects them,
    but individual features can legitimately fail to compute for a given window."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


class StatusStateMachine:
    def __init__(
        self,
        threshold_policy: dict = None,
        hysteresis_windows: int = DEFAULT_HYSTERESIS_WINDOWS,
        stale_gap_s: float = DEFAULT_STALE_GAP_S,
        model_version: str | None = None,
        feature_version: str | None = None,
    ):
        self.threshold_policy = threshold_policy or DEFAULT_THRESHOLD_POLICY
        self.hysteresis_windows = hysteresis_windows
        self.stale_gap_s = stale_gap_s
        self.model_version = model_version
        self.feature_version = feature_version

        self.current_state = WAITING
        self._pending_bucket: str | None = None
        self._pending_count = 0
        self._last_window_end_s: float | None = None

    def _bucket_for(self, probability: float) -> str:
        if probability < self.threshold_policy["green_max"]:
            return GREEN
        if probability < self.threshold_policy["amber_max"]:
            return AMBER
        return RED

    def _apply_hysteresis(self, bucket: str) -> str:
        if bucket == self._pending_bucket:
            self._pending_count += 1
        else:
            self._pending_bucket = bucket
            self._pending_count = 1

        if self.current_state not in COLOR_STATES or self._pending_count >= self.hysteresis_windows:
            self.current_state = bucket
        return self.current_state

    def _reset_hysteresis(self) -> None:
        self._pending_bucket = None
        self._pending_count = 0

    def update(
        self,
        window_start_s: float,
        window_end_s: float,
        probability: float | None = None,
        abstention: AbstentionResult | None = None,
        baseline_ready: bool = True,
        quality: dict | None = None,
    ) -> StatusRecord:
        reasons: list[str] = []

        gap_s = None if self._last_window_end_s is None else window_start_s - self._last_window_end_s
        if gap_s is not None and gap_s > self.stale_gap_s:
            self.current_state = STALE
            self._reset_hysteresis()
            reasons = [f"gap of {gap_s:.0f}s since last update exceeds {self.stale_gap_s}s"]
        elif not baseline_ready:
            self.current_state = CALIBRATING
            self._reset_hysteresis()
        elif abstention is not None and abstention.abstain:
            self.current_state = abstention.reason
            self._reset_hysteresis()
            reasons = list(abstention.triggers)
        elif probability is None:
            self.current_state = WAITING
            self._reset_hysteresis()
        else:
            bucket = self._bucket_for(probability)
            self._apply_hysteresis(bucket)

        self._last_window_end_s = window_end_s

        return StatusRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            state=self.current_state,
            probability=probability,
            model_version=self.model_version,
            feature_version=self.feature_version,
            quality=quality or {},
            reasons=reasons,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )

    def mark_error(self, message: str) -> StatusRecord:
        self.current_state = ERROR
        self._reset_hysteresis()
        return StatusRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            state=ERROR,
            probability=None,
            model_version=self.model_version,
            feature_version=self.feature_version,
            reasons=[message],
        )


def save_status_log(records: list[StatusRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")
