from neuroshield.runtime.dynamics import (
    TREND_FALLING,
    TREND_RISING,
    TREND_STEADY,
    hrv_proxy_recovery,
    recovery_trend,
    session_summary,
    stress_episodes,
    time_in_state,
)
from neuroshield.runtime.status import StatusRecord


def _rec(state, start, end, index=None, rmssd=None):
    return StatusRecord(
        timestamp="t",
        state=state,
        probability=None,
        model_version="m2",
        feature_version="features-v1",
        window_start_s=start,
        window_end_s=end,
        stress_index=index,
        values={"ibi_rmssd_ms": rmssd} if rmssd is not None else {},
    )


class TestTimeInState:
    def test_sums_durations_per_state(self):
        recs = [_rec("green", 0, 60), _rec("green", 30, 90), _rec("red", 60, 120)]
        tis = time_in_state(recs)
        assert tis["green"] == 120.0
        assert tis["red"] == 60.0


class TestRecoveryTrend:
    def test_rising(self):
        recs = [_rec("amber", i * 30, i * 30 + 60, index=v) for i, v in enumerate([10, 30, 50, 70, 90])]
        assert recovery_trend(recs) == TREND_RISING

    def test_falling(self):
        recs = [_rec("amber", i * 30, i * 30 + 60, index=v) for i, v in enumerate([90, 70, 50, 30, 10])]
        assert recovery_trend(recs) == TREND_FALLING

    def test_steady(self):
        recs = [_rec("green", i * 30, i * 30 + 60, index=v) for i, v in enumerate([40, 41, 40, 39, 40])]
        assert recovery_trend(recs) == TREND_STEADY

    def test_too_few_points_is_steady(self):
        assert recovery_trend([_rec("green", 0, 60, index=50)]) == TREND_STEADY


class TestStressEpisodes:
    def test_detects_contiguous_elevated_run(self):
        recs = [
            _rec("green", 0, 60, index=10),
            _rec("amber", 30, 90, index=55),
            _rec("red", 60, 120, index=85),
            _rec("green", 90, 150, index=20),
        ]
        eps = stress_episodes(recs, min_windows=2)
        assert len(eps) == 1
        assert eps[0]["n_windows"] == 2
        assert eps[0]["peak_state"] == "red"
        assert eps[0]["peak_index"] == 85

    def test_short_runs_below_min_are_ignored(self):
        recs = [_rec("green", 0, 60), _rec("red", 30, 90), _rec("green", 60, 120)]
        assert stress_episodes(recs, min_windows=2) == []

    def test_trailing_episode_captured(self):
        recs = [_rec("green", 0, 60, index=5), _rec("amber", 30, 90, index=55), _rec("amber", 60, 120, index=60)]
        eps = stress_episodes(recs, min_windows=2)
        assert len(eps) == 1


class TestHrvProxy:
    def test_mean_of_recent_rmssd(self):
        recs = [_rec("green", 0, 60, rmssd=40.0), _rec("green", 30, 90, rmssd=60.0)]
        assert hrv_proxy_recovery(recs) == 50.0

    def test_none_when_no_rmssd(self):
        assert hrv_proxy_recovery([_rec("green", 0, 60)]) is None


class TestSessionSummary:
    def test_summary_has_all_sections(self):
        recs = [
            _rec("green", 0, 60, index=10, rmssd=45.0),
            _rec("amber", 30, 90, index=55, rmssd=40.0),
            _rec("red", 60, 120, index=85, rmssd=35.0),
        ]
        summary = session_summary(recs)
        for key in ("time_in_state", "recovery_trend", "hrv_proxy_recovery", "episodes", "index_summary"):
            assert key in summary
        assert summary["index_summary"]["max"] == 85
        assert summary["n_scored_windows"] == 3

    def test_json_serializable(self):
        import json

        recs = [_rec("green", 0, 60, index=10, rmssd=45.0), _rec("amber", 30, 90, index=55)]
        json.dumps(session_summary(recs))

    def test_empty_records(self):
        summary = session_summary([])
        assert summary["n_windows"] == 0
        assert summary["index_summary"]["max"] is None
