import json

import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS, FEATURE_VERSION
from neuroshield.models.artifact import IncompatibleFeatureVersionError, MissingFeatureColumnsError
from neuroshield.runtime.baseline import (
    DEFAULT_MIN_STD,
    compute_baseline_from_events,
    compute_baseline_profile,
    load_baseline_profile,
    save_baseline_profile,
    zscore_features,
)
from neuroshield.runtime.synthetic_source import generate_events


def _quiet_windows(n=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {col: rng.normal(10.0, 1.0) for col in FEATURE_COLUMNS}
        row["window_start_s"] = i * 30.0
        row["window_end_s"] = i * 30.0 + 60.0
        row["valid_fraction"] = 1.0
        row["subject_id"] = "s1"
        rows.append(row)
    return pd.DataFrame(rows)


class TestComputeBaselineProfile:
    def test_required_schema_fields_present(self):
        profile = compute_baseline_profile(_quiet_windows(), source="test", subject_id="s1")
        for key in (
            "feature_means",
            "feature_stds",
            "accepted_seconds",
            "source",
            "feature_version",
            "created_at",
            "subject_id",
        ):
            assert key in profile

    def test_means_and_stds_cover_every_feature_column(self):
        profile = compute_baseline_profile(_quiet_windows())
        assert set(profile["feature_means"].keys()) == set(FEATURE_COLUMNS)
        assert set(profile["feature_stds"].keys()) == set(FEATURE_COLUMNS)

    def test_feature_version_matches_extractor(self):
        profile = compute_baseline_profile(_quiet_windows())
        assert profile["feature_version"] == FEATURE_VERSION

    def test_min_std_floor_applied_to_constant_feature(self):
        windows = _quiet_windows()
        windows["hr_mean_bpm"] = 70.0  # zero variance
        profile = compute_baseline_profile(windows, min_std=DEFAULT_MIN_STD)
        assert profile["feature_stds"]["hr_mean_bpm"] == DEFAULT_MIN_STD

    def test_low_valid_fraction_windows_are_excluded(self):
        windows = _quiet_windows(n=10)
        windows.loc[0:4, "valid_fraction"] = 0.1  # half the windows are low quality
        profile = compute_baseline_profile(windows, min_valid_fraction=0.9)
        assert profile["n_accepted_windows"] == 5
        assert profile["n_total_windows"] == 10

    def test_accepted_seconds_spans_accepted_window_range(self):
        windows = _quiet_windows(n=5)  # windows at [0,60],[30,90],[60,120],[90,150],[120,180]
        profile = compute_baseline_profile(windows)
        assert profile["accepted_seconds"] == pytest.approx(180.0)

    def test_raises_when_all_windows_rejected(self):
        windows = _quiet_windows()
        windows["valid_fraction"] = 0.0
        with pytest.raises(ValueError, match="no windows met"):
            compute_baseline_profile(windows, min_valid_fraction=0.9)


class TestSaveLoadRoundtrip:
    def test_round_trip_preserves_profile(self, tmp_path):
        profile = compute_baseline_profile(_quiet_windows(), source="test", subject_id="s1")
        path = tmp_path / "baseline.json"
        save_baseline_profile(profile, path)
        reloaded = load_baseline_profile(path)
        assert reloaded == profile

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline_profile(tmp_path / "nope.json")

    def test_incompatible_feature_version_refuses_to_load(self, tmp_path):
        profile = compute_baseline_profile(_quiet_windows())
        profile["feature_version"] = "features-v0"
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(profile))
        with pytest.raises(IncompatibleFeatureVersionError):
            load_baseline_profile(path)


class TestZscoreFeatures:
    def test_zscore_matches_manual_calculation(self):
        profile = {
            "feature_means": {"hr_mean_bpm": 70.0, "eda_level": 0.3},
            "feature_stds": {"hr_mean_bpm": 5.0, "eda_level": 0.1},
        }
        features = pd.DataFrame({"hr_mean_bpm": [75.0, 65.0], "eda_level": [0.4, 0.2]})
        result = zscore_features(features, profile, feature_columns=["hr_mean_bpm", "eda_level"])
        assert result["hr_mean_bpm_z"].tolist() == pytest.approx([1.0, -1.0])
        assert result["eda_level_z"].tolist() == pytest.approx([1.0, -1.0])

    def test_original_columns_preserved(self):
        profile = {"feature_means": {"hr_mean_bpm": 70.0}, "feature_stds": {"hr_mean_bpm": 5.0}}
        features = pd.DataFrame({"hr_mean_bpm": [75.0], "subject_id": ["s1"]})
        result = zscore_features(features, profile, feature_columns=["hr_mean_bpm"])
        assert "subject_id" in result.columns
        assert "hr_mean_bpm" in result.columns

    def test_missing_column_raises(self):
        profile = {"feature_means": {"hr_mean_bpm": 70.0}, "feature_stds": {"hr_mean_bpm": 5.0}}
        features = pd.DataFrame({"eda_level": [0.3]})
        with pytest.raises(MissingFeatureColumnsError):
            zscore_features(features, profile, feature_columns=["hr_mean_bpm"])

    def test_defaults_to_profiles_own_feature_columns(self):
        profile = {"feature_means": {"hr_mean_bpm": 70.0}, "feature_stds": {"hr_mean_bpm": 5.0}}
        features = pd.DataFrame({"hr_mean_bpm": [80.0]})
        result = zscore_features(features, profile)
        assert result["hr_mean_bpm_z"].iloc[0] == pytest.approx(2.0)


class TestComputeBaselineFromEvents:
    def test_end_to_end_from_synthetic_quiet_segment(self):
        events = generate_events(duration_sec=180.0, seed=9, phases=[("quiet_baseline", 1.0)])
        profile = compute_baseline_from_events(events, source="synthetic", subject_id="s1")
        assert profile["feature_version"] == FEATURE_VERSION
        assert profile["n_accepted_windows"] > 0
        assert profile["accepted_seconds"] > 0
        assert set(profile["feature_means"].keys()) == set(FEATURE_COLUMNS)
