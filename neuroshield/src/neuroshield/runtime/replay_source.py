"""Replay a NDJSON raw event fixture (docs/contracts.md, ``neuroshield.hw.v1``) as typed events.

This is the software stand-in for a live serial/BLE hardware source: it reads events in
timestamp order, validates every one against the contract, and can play them back at real speed,
an accelerated multiple, or as fast as possible. Malformed lines are never silently dropped --
they are preserved in a raw log with a parse error, and every counter T11 asks for (valid,
invalid, missing fields, unknown schema versions, stale periods) is tracked as the stream is
consumed, so a future serial adapter can be held to the exact same accounting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, ValidationError

SCHEMA_VERSION = "neuroshield.hw.v1"
KNOWN_EVENT_TYPES = {"ppg", "eda", "temp", "imu", "health"}
REQUIRED_COMMON_FIELDS = {"schema_version", "type", "source", "session_id", "seq", "t_us", "ok"}

DEFAULT_STALE_GAP_US = 2_000_000  # 2 seconds with no valid event is considered a stale period


class PpgEvent(BaseModel):
    schema_version: str
    type: Literal["ppg"]
    source: str
    session_id: str
    seq: int
    t_us: int
    ok: bool
    ppg_raw: int | None = None
    error: str | None = None


class EdaEvent(BaseModel):
    schema_version: str
    type: Literal["eda"]
    source: str
    session_id: str
    seq: int
    t_us: int
    ok: bool
    eda_level: float | None = None
    eda_unit: str | None = None
    error: str | None = None


class TempEvent(BaseModel):
    schema_version: str
    type: Literal["temp"]
    source: str
    session_id: str
    seq: int
    t_us: int
    ok: bool
    temp_c: float | None = None
    error: str | None = None


class ImuEvent(BaseModel):
    schema_version: str
    type: Literal["imu"]
    source: str
    session_id: str
    seq: int
    t_us: int
    ok: bool
    acc_x: float | None = None
    acc_y: float | None = None
    acc_z: float | None = None
    gyro_x: float | None = None
    gyro_y: float | None = None
    gyro_z: float | None = None
    error: str | None = None


class HealthEvent(BaseModel):
    schema_version: str
    type: Literal["health"]
    source: str
    session_id: str
    seq: int
    t_us: int
    ok: bool
    battery_pct: float | None = None
    channels: dict[str, bool] | None = None
    link_quality: int | None = None
    uptime_s: float | None = None
    fault: str | None = None


EVENT_MODEL_BY_TYPE: dict[str, type[BaseModel]] = {
    "ppg": PpgEvent,
    "eda": EdaEvent,
    "temp": TempEvent,
    "imu": ImuEvent,
    "health": HealthEvent,
}


@dataclass
class ReplayCounters:
    valid_events: int = 0
    invalid_events: int = 0
    missing_fields: int = 0
    unknown_schema_versions: int = 0
    stale_periods: int = 0

    def as_dict(self) -> dict:
        return {
            "valid_events": self.valid_events,
            "invalid_events": self.invalid_events,
            "missing_fields": self.missing_fields,
            "unknown_schema_versions": self.unknown_schema_versions,
            "stale_periods": self.stale_periods,
        }


@dataclass
class RawLogEntry:
    line_number: int
    raw_line: str
    error: str


def _parse_and_validate(line: str, line_number: int) -> tuple[BaseModel | None, RawLogEntry | None, str | None]:
    """Returns (validated_event_or_None, raw_log_entry_or_None, category).

    ``category`` is one of "missing_fields", "unknown_schema_version", "other", or None (valid).
    """
    stripped = line.strip()
    if not stripped:
        return None, None, None  # blank lines are not events, not errors

    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, RawLogEntry(line_number, stripped, f"JSON parse error: {exc}"), "other"

    if not isinstance(raw, dict):
        return None, RawLogEntry(line_number, stripped, "event line is not a JSON object"), "other"

    missing = REQUIRED_COMMON_FIELDS - raw.keys()
    if missing:
        return (
            None,
            RawLogEntry(line_number, stripped, f"missing required fields: {sorted(missing)}"),
            "missing_fields",
        )

    if raw["schema_version"] != SCHEMA_VERSION:
        return (
            None,
            RawLogEntry(
                line_number,
                stripped,
                f"unknown schema_version {raw['schema_version']!r}, expected {SCHEMA_VERSION!r}",
            ),
            "unknown_schema_version",
        )

    event_type = raw.get("type")
    if event_type not in KNOWN_EVENT_TYPES:
        return (
            None,
            RawLogEntry(line_number, stripped, f"unknown event type {event_type!r}"),
            "other",
        )

    model_cls = EVENT_MODEL_BY_TYPE[event_type]
    try:
        event = model_cls.model_validate(raw)
    except ValidationError as exc:
        return None, RawLogEntry(line_number, stripped, f"validation error: {exc}"), "other"

    return event, None, None


class ReplaySource:
    """Iterates a NDJSON fixture once, yielding validated events in file order.

    ``speed``: ``None`` plays back as fast as possible (no pacing, the default -- good for
    batch processing and tests). ``1.0`` paces playback to match the original ``t_us`` deltas in
    real time. Any other positive float accelerates (``10.0`` = 10x faster than real time).
    """

    def __init__(self, path: Path, speed: float | None = None, stale_gap_us: int = DEFAULT_STALE_GAP_US):
        self.path = Path(path)
        self.speed = speed
        self.stale_gap_us = stale_gap_us
        self.counters = ReplayCounters()
        self.raw_log: list[RawLogEntry] = []

    def __iter__(self) -> Iterator[dict]:
        last_valid_t_us: int | None = None
        with open(self.path) as f:
            for line_number, line in enumerate(f, start=1):
                event, log_entry, category = _parse_and_validate(line, line_number)

                if log_entry is not None:
                    self.raw_log.append(log_entry)
                    self.counters.invalid_events += 1
                    if category == "missing_fields":
                        self.counters.missing_fields += 1
                    elif category == "unknown_schema_version":
                        self.counters.unknown_schema_versions += 1
                    continue

                if event is None:
                    continue  # blank line

                self.counters.valid_events += 1

                if last_valid_t_us is not None and event.t_us - last_valid_t_us > self.stale_gap_us:
                    self.counters.stale_periods += 1

                if self.speed is not None and last_valid_t_us is not None:
                    dt_us = event.t_us - last_valid_t_us
                    if dt_us > 0:
                        time.sleep((dt_us / 1_000_000.0) / self.speed)

                last_valid_t_us = event.t_us
                yield event.model_dump()

    def write_raw_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self.raw_log:
                f.write(json.dumps({"line": entry.line_number, "raw": entry.raw_line, "error": entry.error}) + "\n")
