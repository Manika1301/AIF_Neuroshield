"""Session engine: wires source -> features -> baseline -> quality gate -> multi-head model ->
status -> explanations -> enriched payload.

D6/D7 refactor: the engine now drives the multi-head model (Head A graded stress + Head B affect)
and enriches every window into a full KPI record -- stress index (0-100), level, affect state, and
the four physiological axes -- in addition to the traffic-light state. Processing is a per-window
step (``_process_window``); ``iter_status`` exposes it as a generator so the WebSocket can stream
one enriched record per window while the batch path (``run_calibration``) consumes the same steps
to build ``status_history``. The sequence is identical either way.

Replay/synthetic sources are fully available up front, so calibration processes the remainder
eagerly; a future hardware source would feed ``_process_window`` incrementally as samples arrive,
with everything downstream unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuroshield.features.extract import FEATURE_COLUMNS, FEATURE_VERSION, extract_features
from neuroshield.features.personalize import add_personalized_features
from neuroshield.models.multihead import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    load_multihead_artifact,
)
from neuroshield.runtime.axes import compute_axes
from neuroshield.runtime.baseline import compute_baseline_from_events, zscore_features
from neuroshield.runtime.dynamics import session_summary
from neuroshield.runtime.events_to_bundle import events_to_bundle
from neuroshield.runtime.explain import explain_status
from neuroshield.runtime.quality_gate import check_abstention
from neuroshield.runtime.replay_source import SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION
from neuroshield.runtime.replay_source import ReplaySource
from neuroshield.runtime.status import StatusRecord, StatusStateMachine
from neuroshield.runtime.synthetic_source import generate_events

SOURCE_MODES = ("synthetic", "replay", "serial")
IMPLEMENTED_SOURCE_MODES = ("synthetic", "replay")

VALUE_FEATURES = ("hr_mean_bpm", "eda_level", "temp_mean_c", "ibi_rmssd_ms")


class EngineError(Exception):
    error_code: str = "engine_error"
    status_code: int = 400


class MissingModelError(EngineError):
    error_code = "missing_model"
    status_code = 503


class MissingBaselineError(EngineError):
    error_code = "missing_baseline"
    status_code = 409


class SchemaMismatchError(EngineError):
    error_code = "schema_mismatch"
    status_code = 422


class UnsupportedSourceModeError(EngineError):
    error_code = "unsupported_source_mode"
    status_code = 400


class SessionNotStartedError(EngineError):
    error_code = "session_not_started"
    status_code = 409


class RuntimeEngine:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        hysteresis_windows: int = 2,
    ):
        self.model_path = model_path
        self.manifest_path = manifest_path
        self.hysteresis_windows = hysteresis_windows

        self.model = None
        self.manifest = None
        self.model_error: str | None = None
        self._load_model()

        self.session_id: str | None = None
        self.source_mode: str | None = None
        self._events: list[dict] = []
        self.baseline: dict | None = None
        self.status_history: list[StatusRecord] = []
        self.state_machine: StatusStateMachine | None = None

    def _load_model(self) -> None:
        try:
            self.model, self.manifest = load_multihead_artifact(self.model_path, self.manifest_path)
        except Exception as exc:  # noqa: BLE001 - any load failure -> not ready, reported via health
            self.model, self.manifest = None, None
            self.model_error = str(exc)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.manifest is not None

    @property
    def baseline_loaded(self) -> bool:
        return self.baseline is not None

    @property
    def source_connected(self) -> bool:
        return self.session_id is not None

    def start_session(
        self,
        source_mode: str,
        session_id: str,
        replay_path: Path | None = None,
        duration_sec: float = 600.0,
        seed: int = 0,
    ) -> None:
        if source_mode not in SOURCE_MODES:
            raise UnsupportedSourceModeError(f"unknown source_mode {source_mode!r}")
        if source_mode not in IMPLEMENTED_SOURCE_MODES:
            raise UnsupportedSourceModeError(
                f"source_mode {source_mode!r} is not implemented yet (hardware handoff is T20)"
            )

        if source_mode == "synthetic":
            events = generate_events(duration_sec=duration_sec, seed=seed, session_id=session_id)
        else:
            if replay_path is None:
                raise EngineError("replay_path is required for source_mode='replay'")
            replay = ReplaySource(replay_path, speed=None)
            events = list(replay)
            if replay.counters.valid_events == 0:
                raise SchemaMismatchError(
                    f"replay file {replay_path} produced zero valid events "
                    f"(expected schema_version={CONTRACT_SCHEMA_VERSION!r}); check the fixture"
                )

        self.session_id = session_id
        self.source_mode = source_mode
        self._events = events
        self.baseline = None
        self.status_history = []
        self.state_machine = StatusStateMachine(
            threshold_policy=self.manifest["threshold_policy"] if self.manifest else None,
            hysteresis_windows=self.hysteresis_windows,
            model_version=self.manifest["model_version"] if self.manifest else None,
            feature_version=FEATURE_VERSION,
        )

    def run_calibration(self, quiet_seconds: float) -> dict:
        if not self.model_loaded:
            raise MissingModelError(self.model_error or "no multi-head model artifact is loaded")
        if self.session_id is None:
            raise SessionNotStartedError("no session is running; call /session/start first")

        quiet_events = [e for e in self._events if e["t_us"] < quiet_seconds * 1_000_000]
        try:
            self.baseline = compute_baseline_from_events(
                quiet_events, source=self.source_mode, subject_id=self.session_id
            )
        except ValueError as exc:
            raise MissingBaselineError(f"calibration failed: {exc}") from exc

        self.status_history = list(self.iter_status())
        return self.baseline

    def iter_status(self):
        """Yield one enriched StatusRecord per feature window (the streaming step)."""
        bundle = events_to_bundle(self._events, dataset=self.source_mode, subject_id=self.session_id)
        features = extract_features(bundle)
        # The model's personalized half is computed against this session's real calibration profile,
        # exactly the reference the training path derives per subject (features.personalize).
        features = add_personalized_features(features, profile=self.baseline)
        z = zscore_features(features, self.baseline)
        feature_columns = self.manifest["feature_columns"]

        for _, row in z.iterrows():
            yield self._process_window(row, feature_columns)

    def _process_window(self, row, feature_columns: list[str]) -> StatusRecord:
        abstention = check_abstention(row)
        probability = None
        prediction = None
        if not abstention.abstain:
            frame = pd.DataFrame([row[feature_columns]])
            prediction = self.model.predict(frame).iloc[0]
            probability = float(prediction["stress_prob"])

        record = self.state_machine.update(
            window_start_s=row["window_start_s"],
            window_end_s=row["window_end_s"],
            probability=probability,
            abstention=abstention,
            baseline_ready=True,
            quality={
                "valid_fraction": row["valid_fraction"],
                "ppg_quality": row["ppg_quality"],
                "motion_dynamic_rms": row["motion_dynamic_rms"],
                "motion_dynamic_p95": row["motion_dynamic_p95"],
            },
        )

        z_scores = {col: row[f"{col}_z"] for col in FEATURE_COLUMNS}
        record.reasons = explain_status(
            record.state, z_scores=z_scores, coefficients=None, abstention=abstention
        )
        record.values = {f: float(row[f]) for f in VALUE_FEATURES}
        record.axes = compute_axes(z_scores)
        if prediction is not None:
            record.stress_index = int(prediction["stress_index"])
            record.level = str(prediction["level"])
            record.affect_state = None if prediction["affect_state"] is None else str(prediction["affect_state"])
        return record

    def latest_status(self) -> StatusRecord:
        if self.status_history:
            return self.status_history[-1]
        state = "calibrating" if self.source_connected and not self.baseline_loaded else "waiting"
        return StatusRecord(
            timestamp=_now_iso(),
            state=state,
            probability=None,
            model_version=self.manifest["model_version"] if self.manifest else None,
            feature_version=FEATURE_VERSION,
        )

    def history(self, limit: int | None = None) -> list[StatusRecord]:
        if limit is None:
            return list(self.status_history)
        return self.status_history[-limit:]

    def session_summary(self) -> dict:
        """Tier-3 dynamics over the current session's status history."""
        return session_summary(self.status_history)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
