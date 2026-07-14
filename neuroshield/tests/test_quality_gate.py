import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.runtime.events_to_bundle import events_to_bundle
from neuroshield.runtime.quality_gate import (
    MOTION_DYNAMIC_P95_MAX,
    MOTION_DYNAMIC_RMS_MAX,
    MOTION_PAUSED,
    POOR_SIGNAL,
    PPG_QUALITY_MIN,
    VALID_FRACTION_MIN,
    annotate_abstention,
    check_abstention,
)
from neuroshield.features.extract import extract_features
from neuroshield.runtime.synthetic_source import DEFAULT_PHASES, generate_events, resolve_phase_schedule


def _healthy_row(**overrides) -> dict:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row.update(
        {
            "motion_dynamic_rms": 0.05,
            "motion_dynamic_p95": 0.10,
            "ppg_quality": 0.99,
            "valid_fraction": 1.0,
        }
    )
    row.update(overrides)
    return row


class TestCheckAbstentionUnit:
    def test_healthy_row_has_no_abstention(self):
        result = check_abstention(_healthy_row())
        assert result.abstain is False
        assert result.reason is None
        assert result.triggers == []

    def test_high_motion_rms_triggers_motion_paused(self):
        result = check_abstention(_healthy_row(motion_dynamic_rms=MOTION_DYNAMIC_RMS_MAX + 1.0))
        assert result.abstain is True
        assert result.reason == MOTION_PAUSED

    def test_high_motion_p95_triggers_motion_paused(self):
        result = check_abstention(_healthy_row(motion_dynamic_p95=MOTION_DYNAMIC_P95_MAX + 1.0))
        assert result.reason == MOTION_PAUSED

    def test_low_ppg_quality_triggers_poor_signal(self):
        result = check_abstention(_healthy_row(ppg_quality=PPG_QUALITY_MIN - 0.1))
        assert result.abstain is True
        assert result.reason == POOR_SIGNAL

    def test_low_valid_fraction_triggers_poor_signal(self):
        result = check_abstention(_healthy_row(valid_fraction=VALID_FRACTION_MIN - 0.1))
        assert result.reason == POOR_SIGNAL

    def test_nan_motion_triggers_poor_signal_not_motion_paused(self):
        result = check_abstention(_healthy_row(motion_dynamic_rms=np.nan, motion_dynamic_p95=np.nan))
        assert result.reason == POOR_SIGNAL

    def test_motion_takes_precedence_over_poor_signal(self):
        result = check_abstention(
            _healthy_row(motion_dynamic_rms=MOTION_DYNAMIC_RMS_MAX + 1.0, ppg_quality=0.0)
        )
        assert result.reason == MOTION_PAUSED

    def test_triggers_list_is_populated_and_descriptive(self):
        result = check_abstention(_healthy_row(motion_dynamic_rms=MOTION_DYNAMIC_RMS_MAX + 1.0))
        assert len(result.triggers) >= 1
        assert "motion_dynamic_rms" in result.triggers[0]


class TestAnnotateAbstention:
    def test_adds_expected_columns(self):
        df = pd.DataFrame([_healthy_row(), _healthy_row(motion_dynamic_rms=5.0)])
        out = annotate_abstention(df)
        assert list(out["abstain"]) == [False, True]
        assert list(out["abstention_reason"]) == [None, MOTION_PAUSED]

    def test_preserves_original_columns(self):
        df = pd.DataFrame([_healthy_row()])
        out = annotate_abstention(df)
        assert "motion_dynamic_rms" in out.columns


class TestSyntheticMotionBurstIntegration:
    """T13 done criteria: replay visibly pauses during motion and resumes after recovery, and no
    low-quality window ever gets a confident status (checked further in T14)."""

    @staticmethod
    @pytest.fixture(scope="class")
    def annotated():
        events = generate_events(duration_sec=600.0, seed=7, session_id="demo-001", phases=DEFAULT_PHASES)
        bundle = events_to_bundle(events)
        features = extract_features(bundle)
        return annotate_abstention(features), resolve_phase_schedule(600.0, DEFAULT_PHASES)

    def test_motion_burst_windows_are_paused(self, annotated):
        features, phases = annotated
        motion_phase = next(p for p in phases if p.name == "motion_burst")
        # A window is "within" the motion phase if it starts and ends inside it.
        in_motion = features[
            (features["window_start_s"] >= motion_phase.start_s) & (features["window_end_s"] <= motion_phase.end_s)
        ]
        assert len(in_motion) > 0
        assert (in_motion["abstention_reason"] == MOTION_PAUSED).all()

    def test_quiet_baseline_windows_are_not_paused(self, annotated):
        features, phases = annotated
        quiet_phase = next(p for p in phases if p.name == "quiet_baseline")
        in_quiet = features[
            (features["window_start_s"] >= quiet_phase.start_s) & (features["window_end_s"] <= quiet_phase.end_s)
        ]
        assert len(in_quiet) > 0
        assert (in_quiet["abstain"] == False).all()  # noqa: E712

    def test_resumes_after_motion_during_recovery(self, annotated):
        features, phases = annotated
        recovery_phase = next(p for p in phases if p.name == "recovery")
        # Skip the very first recovery window, which may straddle the motion_burst boundary.
        in_recovery = features[
            (features["window_start_s"] > recovery_phase.start_s + 30)
            & (features["window_end_s"] <= recovery_phase.end_s)
        ]
        assert len(in_recovery) > 0
        assert (in_recovery["abstain"] == False).all()  # noqa: E712

    def test_sensor_fault_windows_are_poor_signal(self, annotated):
        features, phases = annotated
        fault_phase = next(p for p in phases if p.name == "sensor_fault")
        in_fault = features[
            (features["window_start_s"] >= fault_phase.start_s) & (features["window_end_s"] <= fault_phase.end_s)
        ]
        assert len(in_fault) > 0
        assert (in_fault["abstention_reason"] == POOR_SIGNAL).all()

    def test_no_abstaining_window_would_be_mistaken_for_confident_status(self, annotated):
        features, _ = annotated
        abstaining = features[features["abstain"]]
        assert len(abstaining) > 0
        assert abstaining["abstention_reason"].isin([MOTION_PAUSED, POOR_SIGNAL]).all()
