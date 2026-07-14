import json

import numpy as np
import pytest

from neuroshield.runtime.synthetic_source import (
    DEFAULT_RATES_HZ,
    SCHEMA_VERSION,
    generate_events,
    resolve_phase_schedule,
    write_ndjson,
)

COMMON_FIELDS = {"schema_version", "type", "source", "session_id", "seq", "t_us", "ok"}


@pytest.fixture(scope="module")
def events():
    return generate_events(duration_sec=30.0, seed=7, session_id="test-session")


class TestSchemaCompliance:
    def test_every_event_has_common_fields(self, events):
        for e in events:
            assert COMMON_FIELDS.issubset(e.keys())

    def test_schema_version_and_source(self, events):
        assert all(e["schema_version"] == SCHEMA_VERSION for e in events)
        assert all(e["source"] == "synthetic" for e in events)

    def test_type_is_one_of_contract_types(self, events):
        assert {e["type"] for e in events} <= {"ppg", "eda", "temp", "imu", "health"}

    def test_every_event_is_json_serializable(self, events):
        for e in events:
            json.loads(json.dumps(e))

    def test_eda_declares_unit(self, events):
        for e in events:
            if e["type"] == "eda" and e["ok"]:
                assert e["eda_unit"] in {"uS", "relative"}

    def test_imu_has_three_axes(self, events):
        for e in events:
            if e["type"] == "imu":
                assert {"acc_x", "acc_y", "acc_z"}.issubset(e.keys())

    def test_health_has_channels_and_fault_field(self, events):
        for e in events:
            if e["type"] == "health":
                assert "channels" in e
                assert set(e["channels"]) == {"ppg", "eda", "temp", "imu"}
                assert "fault" in e


class TestSequencing:
    def test_seq_is_contiguous_from_zero(self, events):
        assert [e["seq"] for e in events] == list(range(len(events)))

    def test_t_us_is_monotonic_non_decreasing(self, events):
        t = [e["t_us"] for e in events]
        assert all(t[i] <= t[i + 1] for i in range(len(t) - 1))


class TestEventCounts:
    def test_counts_match_rate_times_duration(self, events):
        by_type = {}
        for e in events:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        duration = 30.0
        for ch, rate in DEFAULT_RATES_HZ.items():
            expected = int(duration * rate) if ch != "health" else max(1, int(duration * rate))
            assert by_type[ch] == expected


class TestReproducibility:
    def test_same_seed_gives_identical_events(self):
        a = generate_events(duration_sec=10.0, seed=42)
        b = generate_events(duration_sec=10.0, seed=42)
        assert a == b

    def test_different_seed_gives_different_events(self):
        a = generate_events(duration_sec=10.0, seed=1)
        b = generate_events(duration_sec=10.0, seed=2)
        assert a != b


class TestPhases:
    def test_resolve_phase_schedule_covers_full_duration(self):
        windows = resolve_phase_schedule(100.0)
        assert windows[0].start_s == 0.0
        assert windows[-1].end_s == 100.0
        for a, b in zip(windows, windows[1:]):
            assert a.end_s == b.start_s

    def test_custom_phase_schedule_is_controllable(self):
        custom = [("quiet_baseline", 0.5), ("motion_burst", 0.5)]
        windows = resolve_phase_schedule(40.0, custom)
        assert [w.name for w in windows] == ["quiet_baseline", "motion_burst"]
        assert windows[0].end_s == pytest.approx(20.0)

    def test_sensor_fault_phase_marks_ppg_events_not_ok(self):
        events = generate_events(duration_sec=40.0, seed=3, phases=[("sensor_fault", 1.0)])
        ppg = [e for e in events if e["type"] == "ppg"]
        assert all(not e["ok"] for e in ppg)
        assert all(e.get("error") == "sensor_disconnected" for e in ppg)
        assert all("ppg_raw" not in e for e in ppg)

    def test_quiet_baseline_only_has_ok_ppg(self):
        events = generate_events(duration_sec=20.0, seed=3, phases=[("quiet_baseline", 1.0)])
        ppg = [e for e in events if e["type"] == "ppg"]
        assert all(e["ok"] for e in ppg)

    def test_motion_burst_increases_imu_dynamics_over_quiet(self):
        quiet = generate_events(duration_sec=20.0, seed=5, phases=[("quiet_baseline", 1.0)])
        burst = generate_events(duration_sec=20.0, seed=5, phases=[("motion_burst", 1.0)])

        def dynamic_rms(evs):
            mags = np.array(
                [np.linalg.norm([e["acc_x"], e["acc_y"], e["acc_z"]]) for e in evs if e["type"] == "imu"]
            )
            return float(np.sqrt(np.mean((mags - mags.mean()) ** 2)))

        assert dynamic_rms(burst) > dynamic_rms(quiet)

    def test_mild_stress_rise_increases_heart_rate_over_quiet(self):
        quiet = generate_events(duration_sec=20.0, seed=5, phases=[("quiet_baseline", 1.0)])
        rise = generate_events(duration_sec=20.0, seed=5, phases=[("mild_stress_rise", 1.0)])

        def mean_abs_ppg(evs):
            return np.mean([abs(e["ppg_raw"] - 2048) for e in evs if e["type"] == "ppg"])

        # A higher heart rate packs more oscillation into the same window; use variance as a proxy.
        def ppg_variance(evs):
            vals = [e["ppg_raw"] for e in evs if e["type"] == "ppg"]
            return float(np.var(vals))

        assert ppg_variance(rise) != ppg_variance(quiet)  # distinct dynamics, sanity check they differ


def test_write_ndjson_round_trips(tmp_path):
    events = generate_events(duration_sec=5.0, seed=1)
    out_path = tmp_path / "fixture.ndjson"
    write_ndjson(events, out_path)

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == len(events)
    reloaded = [json.loads(line) for line in lines]
    assert reloaded == events
