import json

import pandas as pd
import pytest

from neuroshield.features.extract import extract_features
from neuroshield.models.artifact import train_final_m1
from neuroshield.runtime.baseline import compute_baseline_from_events
from neuroshield.runtime.events_to_bundle import events_to_bundle
from neuroshield.runtime.quality_gate import MOTION_PAUSED, check_abstention
from neuroshield.runtime.status import (
    AMBER,
    CALIBRATING,
    GREEN,
    RED,
    STALE,
    WAITING,
    StatusStateMachine,
    save_status_log,
)
from neuroshield.runtime.synthetic_source import generate_events, resolve_phase_schedule


class TestBucketingAndHysteresis:
    def test_first_reading_commits_immediately(self):
        sm = StatusStateMachine(hysteresis_windows=3)
        record = sm.update(0.0, 60.0, probability=0.1)
        assert record.state == GREEN

    def test_single_spike_does_not_flip_established_color(self):
        sm = StatusStateMachine(hysteresis_windows=2)
        sm.update(0.0, 60.0, probability=0.1)  # green, committed (first reading)
        sm.update(60.0, 120.0, probability=0.1)  # still green
        spike = sm.update(120.0, 180.0, probability=0.9)  # one red-bucket spike
        assert spike.state == GREEN  # hysteresis holds the previous color

    def test_sustained_change_does_flip_after_hysteresis_count(self):
        sm = StatusStateMachine(hysteresis_windows=2)
        sm.update(0.0, 60.0, probability=0.1)  # green
        sm.update(60.0, 120.0, probability=0.1)  # green
        sm.update(120.0, 180.0, probability=0.9)  # spike 1, still green
        record = sm.update(180.0, 240.0, probability=0.9)  # spike 2, now commits
        assert record.state == RED

    def test_thresholds_partition_green_amber_red(self):
        sm = StatusStateMachine()
        assert sm.update(0.0, 60.0, probability=0.10).state == GREEN
        sm2 = StatusStateMachine()
        assert sm2.update(0.0, 60.0, probability=0.55).state == AMBER
        sm3 = StatusStateMachine()
        assert sm3.update(0.0, 60.0, probability=0.85).state == RED

    def test_custom_threshold_policy_is_respected(self):
        sm = StatusStateMachine(threshold_policy={"green_max": 0.2, "amber_max": 0.4})
        assert sm.update(0.0, 60.0, probability=0.3).state == AMBER


class TestAbstentionOverridesColor:
    def test_abstention_overrides_immediately_even_mid_color_run(self):
        sm = StatusStateMachine(hysteresis_windows=5)
        sm.update(0.0, 60.0, probability=0.1)
        result = check_abstention(
            {"motion_dynamic_rms": 5.0, "motion_dynamic_p95": 8.0, "ppg_quality": 0.9, "valid_fraction": 1.0}
        )
        record = sm.update(60.0, 120.0, probability=0.9, abstention=result)
        assert record.state == MOTION_PAUSED
        assert record.reasons  # triggers preserved

    def test_color_hysteresis_resets_after_abstention(self):
        sm = StatusStateMachine(hysteresis_windows=2)
        sm.update(0.0, 60.0, probability=0.1)  # green
        motion = check_abstention(
            {"motion_dynamic_rms": 5.0, "motion_dynamic_p95": 8.0, "ppg_quality": 0.9, "valid_fraction": 1.0}
        )
        sm.update(60.0, 120.0, probability=0.9, abstention=motion)  # motion_paused
        # First color reading after abstention commits immediately (not held by stale pending state)
        record = sm.update(120.0, 180.0, probability=0.9)
        assert record.state == RED


class TestCalibratingAndWaiting:
    def test_baseline_not_ready_yields_calibrating(self):
        sm = StatusStateMachine()
        record = sm.update(0.0, 60.0, probability=0.1, baseline_ready=False)
        assert record.state == CALIBRATING

    def test_no_probability_and_no_abstention_yields_waiting(self):
        sm = StatusStateMachine()
        record = sm.update(0.0, 60.0, probability=None)
        assert record.state == WAITING

    def test_initial_state_before_any_update_is_waiting(self):
        sm = StatusStateMachine()
        assert sm.current_state == WAITING


class TestStaleDetection:
    def test_large_gap_between_windows_yields_stale(self):
        sm = StatusStateMachine(stale_gap_s=30.0)
        sm.update(0.0, 60.0, probability=0.1)
        record = sm.update(200.0, 260.0, probability=0.1)  # 140s gap since last window ended
        assert record.state == STALE

    def test_stale_resets_hysteresis(self):
        sm = StatusStateMachine(hysteresis_windows=2, stale_gap_s=30.0)
        sm.update(0.0, 60.0, probability=0.1)
        sm.update(200.0, 260.0, probability=0.1)  # stale
        record = sm.update(260.0, 320.0, probability=0.9)  # first reading after stale
        assert record.state == RED  # commits immediately, current_state wasn't a color state


class TestStatusRecordContents:
    def test_record_has_required_fields(self):
        sm = StatusStateMachine(model_version="m1_v1", feature_version="features-v1")
        record = sm.update(0.0, 60.0, probability=0.3, quality={"valid_fraction": 1.0})
        d = record.to_dict()
        for key in ("timestamp", "state", "probability", "model_version", "feature_version", "quality", "reasons"):
            assert key in d
        assert d["model_version"] == "m1_v1"
        assert d["feature_version"] == "features-v1"

    def test_timestamp_is_parseable_iso8601(self):
        from datetime import datetime

        sm = StatusStateMachine()
        record = sm.update(0.0, 60.0, probability=0.1)
        datetime.fromisoformat(record.timestamp)

    def test_to_dict_sanitizes_nan_to_none_for_strict_json(self):
        import json

        sm = StatusStateMachine()
        record = sm.update(
            0.0,
            60.0,
            probability=0.1,
            quality={"ppg_quality": float("nan")},
        )
        record.values = {"hr_mean_bpm": float("nan"), "eda_level": 0.3}
        d = record.to_dict()
        assert d["quality"]["ppg_quality"] is None
        assert d["values"]["hr_mean_bpm"] is None
        assert d["values"]["eda_level"] == 0.3
        json.dumps(d, allow_nan=False)  # must not raise -- this is what Starlette's JSONResponse does


def test_save_status_log_round_trips(tmp_path):
    sm = StatusStateMachine()
    records = [sm.update(i * 60.0, i * 60.0 + 60.0, probability=0.1) for i in range(3)]
    path = tmp_path / "status_log.ndjson"
    save_status_log(records, path)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert "state" in record


class TestFullReplaySequence:
    """T14 done criteria: a deterministic replay produces a predictable state sequence:
    waiting, green, motion paused, recovery, amber/red test segment, recovery."""

    @staticmethod
    @pytest.fixture(scope="class")
    def state_sequence():
        phases = [
            ("quiet_baseline", 0.20),
            ("motion_burst", 0.15),
            ("recovery", 0.15),
            ("mild_stress_rise", 0.20),
            ("recovery", 0.30),
        ]
        duration_sec = 600.0
        events = generate_events(duration_sec=duration_sec, seed=11, session_id="t14-demo", phases=phases)
        phase_bounds = resolve_phase_schedule(duration_sec, phases)

        # Calibrate a baseline from the quiet segment only (T12 integration -- not otherwise used
        # in this state-sequence test, but proves the pipeline's calibration step runs cleanly).
        quiet = phase_bounds[0]
        quiet_events = [e for e in events if e["t_us"] < quiet.end_s * 1_000_000]
        baseline = compute_baseline_from_events(quiet_events, source="synthetic", subject_id="t14")
        assert baseline["n_accepted_windows"] > 0

        bundle = events_to_bundle(events)
        features = extract_features(bundle)

        # Fit a small M1-shaped model on synthetic quiet-vs-stress feature rows so probabilities
        # actually track the driving stress curve, the same technique used in T8/T9's own tests.
        import numpy as np

        rng = np.random.default_rng(0)
        train_rows = []
        for label, shift in ((0, 0.0), (1, 3.0)):
            for _ in range(60):
                row = {col: rng.normal(0, 1) for col in features.columns if col not in (
                    "dataset", "subject_id", "window_start_s", "window_end_s", "label", "feature_version"
                )}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                row["m1_label"] = label
                train_rows.append(row)
        train_df = pd.DataFrame(train_rows)
        from neuroshield.features.extract import FEATURE_COLUMNS

        pipeline = train_final_m1(train_df, feature_columns=FEATURE_COLUMNS, random_state=0)

        sm = StatusStateMachine(model_version="test", feature_version="features-v1", hysteresis_windows=1)
        sequence = []
        for _, row in features.iterrows():
            abst = check_abstention(row)
            prob = None
            if not abst.abstain:
                X = row[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
                prob = float(pipeline.predict_proba(X)[0, 1])
            record = sm.update(row["window_start_s"], row["window_end_s"], probability=prob, abstention=abst)
            sequence.append(record.state)
        return sequence, phase_bounds

    def test_sequence_starts_green_during_quiet(self, state_sequence):
        sequence, _ = state_sequence
        assert sequence[0] == GREEN

    def test_sequence_includes_motion_paused(self, state_sequence):
        sequence, _ = state_sequence
        assert MOTION_PAUSED in sequence

    def test_sequence_recovers_to_green_after_motion(self, state_sequence):
        sequence, _ = state_sequence
        first_motion_idx = sequence.index(MOTION_PAUSED)
        after_motion = sequence[first_motion_idx:]
        assert GREEN in after_motion[after_motion.index(MOTION_PAUSED):]

    def test_sequence_includes_amber_or_red_during_stress_rise(self, state_sequence):
        sequence, _ = state_sequence
        assert AMBER in sequence or RED in sequence

    def test_sequence_ends_back_in_green_recovery(self, state_sequence):
        sequence, _ = state_sequence
        assert sequence[-1] == GREEN

    def test_deduplicated_sequence_matches_expected_milestones(self, state_sequence):
        sequence, _ = state_sequence
        deduped = [s for i, s in enumerate(sequence) if i == 0 or s != sequence[i - 1]]
        # green (quiet) -> motion_paused (burst) -> green (recovery) -> amber/red (rise) -> green (recovery)
        assert deduped[0] == GREEN
        assert MOTION_PAUSED in deduped
        assert (AMBER in deduped) or (RED in deduped)
        assert deduped[-1] == GREEN
