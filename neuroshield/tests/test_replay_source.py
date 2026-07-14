import json
import time
from pathlib import Path

import pytest

from neuroshield.runtime.replay_source import ReplaySource
from neuroshield.runtime.synthetic_source import generate_events, write_ndjson


def _write_fixture(tmp_path, duration_sec=3.0, seed=1, name="fixture.ndjson"):
    events = generate_events(duration_sec=duration_sec, seed=seed)
    path = tmp_path / name
    write_ndjson(events, path)
    return path, events


class TestValidReplay:
    def test_replay_is_deterministic_across_runs(self, tmp_path):
        path, _ = _write_fixture(tmp_path)
        first = list(ReplaySource(path))
        second = list(ReplaySource(path))
        assert first == second

    def test_all_lines_counted_as_valid(self, tmp_path):
        path, events = _write_fixture(tmp_path)
        source = ReplaySource(path)
        yielded = list(source)
        assert len(yielded) == len(events)
        assert source.counters.valid_events == len(events)
        assert source.counters.invalid_events == 0
        assert source.raw_log == []

    def test_events_yielded_in_file_order(self, tmp_path):
        path, events = _write_fixture(tmp_path)
        yielded = list(ReplaySource(path))
        assert [e["seq"] for e in yielded] == [e["seq"] for e in events]

    def test_t_us_non_decreasing(self, tmp_path):
        path, _ = _write_fixture(tmp_path)
        yielded = list(ReplaySource(path))
        t = [e["t_us"] for e in yielded]
        assert all(t[i] <= t[i + 1] for i in range(len(t) - 1))


class TestMalformedLines:
    def _mixed_fixture(self, tmp_path):
        events = generate_events(duration_sec=1.0, seed=2)
        lines = [json.dumps(e) for e in events[:5]]
        valid_written = len(lines)
        lines.append("{not valid json")
        lines.append('{"schema_version": "neuroshield.hw.v1", "type": "ppg"}')  # missing fields
        lines.append(
            '{"schema_version": "neuroshield.hw.v0", "type": "ppg", "source": "synthetic", '
            '"session_id": "s", "seq": 0, "t_us": 0, "ok": true, "ppg_raw": 1}'
        )  # unknown schema version
        lines.append("")  # blank line, should not count as an error
        path = tmp_path / "malformed.ndjson"
        path.write_text("\n".join(lines) + "\n")
        return path, valid_written

    def test_malformed_lines_are_preserved_in_raw_log(self, tmp_path):
        path, valid_written = self._mixed_fixture(tmp_path)
        source = ReplaySource(path)
        yielded = list(source)

        assert len(yielded) == valid_written
        assert source.counters.valid_events == valid_written
        assert source.counters.invalid_events == 3
        assert source.counters.missing_fields == 1
        assert source.counters.unknown_schema_versions == 1
        assert len(source.raw_log) == 3

    def test_blank_lines_are_not_counted_as_errors(self, tmp_path):
        path, _ = self._mixed_fixture(tmp_path)
        source = ReplaySource(path)
        list(source)
        assert not any(entry.raw_line == "" for entry in source.raw_log)

    def test_error_messages_are_stable_across_runs(self, tmp_path):
        path, _ = self._mixed_fixture(tmp_path)
        source_a = ReplaySource(path)
        list(source_a)
        source_b = ReplaySource(path)
        list(source_b)
        assert [e.error for e in source_a.raw_log] == [e.error for e in source_b.raw_log]

    def test_raw_log_preserves_original_line_text(self, tmp_path):
        path, _ = self._mixed_fixture(tmp_path)
        source = ReplaySource(path)
        list(source)
        assert "{not valid json" in [e.raw_line for e in source.raw_log]

    def test_write_raw_log_round_trips(self, tmp_path):
        path, _ = self._mixed_fixture(tmp_path)
        source = ReplaySource(path)
        list(source)
        log_path = tmp_path / "raw_errors.ndjson"
        source.write_raw_log(log_path)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == len(source.raw_log)
        for line in lines:
            record = json.loads(line)
            assert {"line", "raw", "error"}.issubset(record.keys())


class TestStalePeriods:
    def test_gap_beyond_threshold_counts_as_stale(self, tmp_path):
        events = generate_events(duration_sec=0.1, seed=3, session_id="s")
        e0 = next(e for e in events if e["type"] == "health")
        e0 = dict(e0, seq=0, t_us=0)
        e1 = dict(e0, seq=1, t_us=5_000_000)  # 5s later -- beyond the default 2s stale threshold
        path = tmp_path / "gap.ndjson"
        path.write_text(json.dumps(e0) + "\n" + json.dumps(e1) + "\n")

        source = ReplaySource(path, stale_gap_us=2_000_000)
        list(source)
        assert source.counters.stale_periods == 1

    def test_no_gap_no_stale_periods(self, tmp_path):
        path, _ = _write_fixture(tmp_path, duration_sec=2.0)
        source = ReplaySource(path)
        list(source)
        assert source.counters.stale_periods == 0


class TestPacing:
    def test_speed_none_is_fast(self, tmp_path):
        path, _ = _write_fixture(tmp_path, duration_sec=2.0)
        start = time.monotonic()
        list(ReplaySource(path, speed=None))
        elapsed = time.monotonic() - start
        assert elapsed < 1.0

    def test_higher_speed_multiplier_is_faster_than_real_time(self, tmp_path):
        events = generate_events(duration_sec=0.1, seed=4)
        e0 = next(e for e in events if e["type"] == "health")
        health_events = [
            dict(e0, seq=i, t_us=i * 50_000) for i in range(3)
        ]  # 3 events, 50ms apart
        path = tmp_path / "pacing.ndjson"
        path.write_text("\n".join(json.dumps(e) for e in health_events) + "\n")

        start = time.monotonic()
        list(ReplaySource(path, speed=1.0))
        real_elapsed = time.monotonic() - start

        start = time.monotonic()
        list(ReplaySource(path, speed=50.0))
        fast_elapsed = time.monotonic() - start

        assert fast_elapsed < real_elapsed


def test_replaying_the_committed_demo_fixture_has_no_errors():
    fixture = Path("data/fixtures/calm_motion_stress.ndjson")
    if not fixture.exists():
        pytest.skip("committed demo fixture not present")
    source = ReplaySource(fixture)
    events = list(source)
    assert len(events) == source.counters.valid_events
    assert source.counters.invalid_events == 0
    assert source.raw_log == []
