
from neuroshield.runtime.axes import (
    AXIS_NAMES,
    ELEVATED_Z,
    HIGH_Z,
    LEVEL_ELEVATED,
    LEVEL_HIGH,
    LEVEL_NORMAL,
    SCORE_CLIP,
    compute_axes,
    compute_axis,
)


class TestArousalDirection:
    def test_high_heart_rate_raises_cardiac_axis(self):
        axis = compute_axis("cardiac", {"hr_mean_bpm": 3.0, "ibi_sd_ms": 0.0, "ibi_rmssd_ms": 0.0})
        assert axis.score > 0

    def test_high_hrv_lowers_cardiac_axis(self):
        # Higher pulse variability (rmssd) means LESS arousal -> negative contribution.
        axis = compute_axis("cardiac", {"hr_mean_bpm": 0.0, "ibi_sd_ms": 3.0, "ibi_rmssd_ms": 3.0})
        assert axis.score < 0

    def test_cooler_skin_raises_thermal_arousal(self):
        # temp below baseline (negative z) -> arousal (vasoconstriction), so axis score positive.
        axis = compute_axis("thermal", {"temp_mean_c": -2.0, "temp_slope_c_per_min": -2.0})
        assert axis.score > 0

    def test_higher_eda_raises_electrodermal_axis(self):
        axis = compute_axis(
            "electrodermal",
            {"eda_level": 2.0, "eda_slope": 2.0, "eda_response_count": 2.0, "eda_response_mean_amp": 2.0},
        )
        assert axis.score > 0

    def test_more_motion_raises_movement_axis(self):
        axis = compute_axis("movement", {"motion_dynamic_rms": 3.0, "motion_dynamic_p95": 3.0})
        assert axis.score > 0


class TestLevels:
    def test_normal_elevated_high_thresholds(self):
        low = compute_axis("cardiac", {"hr_mean_bpm": 0.5})
        mid = compute_axis("cardiac", {"hr_mean_bpm": ELEVATED_Z + 0.1})
        high = compute_axis("cardiac", {"hr_mean_bpm": HIGH_Z + 0.1})
        assert low.level == LEVEL_NORMAL
        assert mid.level == LEVEL_ELEVATED
        assert high.level == LEVEL_HIGH


class TestNaNSafety:
    def test_missing_features_are_skipped(self):
        axis = compute_axis("cardiac", {"hr_mean_bpm": 2.0})  # only 1 of 3 features present
        assert axis.n_features == 1
        assert axis.score == 2.0

    def test_all_missing_yields_none_score(self):
        axis = compute_axis("cardiac", {})
        assert axis.score is None
        assert axis.level == LEVEL_NORMAL
        assert axis.n_features == 0

    def test_nan_values_are_ignored(self):
        axis = compute_axis("cardiac", {"hr_mean_bpm": float("nan"), "ibi_rmssd_ms": -2.0})
        assert axis.n_features == 1
        # rmssd sign is -1, z is -2 -> contribution +2
        assert axis.score == 2.0


class TestScoreClipping:
    def test_extreme_score_is_clipped(self):
        axis = compute_axis("movement", {"motion_dynamic_rms": 100.0, "motion_dynamic_p95": 100.0})
        assert axis.score == SCORE_CLIP


class TestComputeAxes:
    def test_returns_all_four_axes(self):
        axes = compute_axes({"hr_mean_bpm": 1.0})
        assert set(axes.keys()) == set(AXIS_NAMES)
        for axis in axes.values():
            assert set(axis.keys()) == {"score", "level", "n_features"}

    def test_accepts_z_suffixed_keys(self):
        with_suffix = compute_axes({"hr_mean_bpm_z": 3.0})
        without_suffix = compute_axes({"hr_mean_bpm": 3.0})
        assert with_suffix["cardiac"]["score"] == without_suffix["cardiac"]["score"]

    def test_json_serializable(self):
        import json

        axes = compute_axes({"hr_mean_bpm": 1.5, "eda_level": 2.0, "temp_mean_c": -1.0})
        json.dumps(axes)  # must not raise

    def test_independent_axes(self):
        # Elevated cardiac should not force electrodermal to be elevated.
        axes = compute_axes({"hr_mean_bpm": 3.0, "eda_level": 0.0})
        assert axes["cardiac"]["level"] == LEVEL_HIGH
        assert axes["electrodermal"]["level"] == LEVEL_NORMAL
