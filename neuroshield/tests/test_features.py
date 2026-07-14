import neurokit2 as nk
import numpy as np
import pytest

from neuroshield.data.bundle import from_channel_arrays
from neuroshield.features.extract import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    METADATA_COLUMNS,
    extract_features,
)

DURATION_S = 151.0  # last-sample time is slightly under duration; +1s headroom keeps the
# 90-150s window fully covered, giving 4 windows at 0, 30, 60, 90 (window_sec=60, step_sec=30)
RATES = {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}


@pytest.fixture(scope="module")
def synthetic_bundle():
    n_bvp = int(DURATION_S * RATES["BVP"])
    n_eda = int(DURATION_S * RATES["EDA"])
    n_acc = int(DURATION_S * RATES["ACC"])

    bvp = nk.ppg_simulate(
        duration=int(DURATION_S), sampling_rate=int(RATES["BVP"]), heart_rate=72, random_state=7
    )
    eda = nk.eda_simulate(
        duration=int(DURATION_S), sampling_rate=int(RATES["EDA"]), scr_number=5, random_state=7
    )
    rng = np.random.default_rng(7)
    temp = 33.0 + np.cumsum(rng.normal(0, 0.001, size=n_eda))  # slow drifting temperature
    acc = rng.normal(0, 0.3, size=(n_acc, 3)) + np.array([0.0, 9.8, 0.0])  # gravity on y + small jitter

    channels = {"BVP": bvp[:n_bvp], "EDA": eda[:n_eda], "TEMP": temp, "ACC": acc}
    # Label the first half baseline (1), second half stress (2), per-channel at each channel's own rate.
    labels = {}
    for name, arr in channels.items():
        n = len(arr)
        lab = np.ones(n, dtype=np.int64)
        lab[n // 2 :] = 2
        labels[name] = lab

    return from_channel_arrays(
        dataset="synthetic",
        subject_id="S99",
        channels=channels,
        sample_rates_hz=RATES,
        labels=labels,
    )


class TestFeatureSchema:
    def test_fixed_column_order(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        assert list(df.columns) == ALL_COLUMNS
        assert list(df.columns) == METADATA_COLUMNS + FEATURE_COLUMNS

    def test_feature_version_constant(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        assert (df["feature_version"] == FEATURE_VERSION).all()

    def test_no_all_null_feature_column(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        for col in FEATURE_COLUMNS:
            assert df[col].notna().any(), f"column {col!r} is entirely null across all windows"

    def test_one_row_per_valid_window(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        # 150s duration, 60s window, 30s step -> starts at 0, 30, 60, 90 (90+60=150 fits exactly)
        assert len(df) == 4
        assert list(df["window_start_s"]) == [0.0, 30.0, 60.0, 90.0]

    def test_labels_attached(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        assert df["label"].notna().all()
        assert set(df["label"].unique()) <= {1.0, 2.0}

    def test_subject_and_dataset_metadata(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        assert (df["subject_id"] == "S99").all()
        assert (df["dataset"] == "synthetic").all()

    def test_valid_fraction_in_unit_range(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        assert df["valid_fraction"].between(0.0, 1.0).all()

    def test_plausible_heart_rate(self, synthetic_bundle):
        df = extract_features(synthetic_bundle)
        # simulated at 72 bpm; allow generous tolerance for peak-detection noise
        assert df["hr_mean_bpm"].dropna().between(40, 140).all()

    def test_empty_window_is_skipped_not_fabricated(self):
        channels = {"EDA": np.array([1.0, 2.0, 3.0])}
        labels = {"EDA": np.array([1, 1, 1])}
        bundle = from_channel_arrays(
            dataset="synthetic",
            subject_id="S100",
            channels=channels,
            sample_rates_hz={"EDA": 4.0},
            labels=labels,
        )
        df = extract_features(bundle, window_sec=60.0, step_sec=30.0)
        assert len(df) == 0


def test_all_columns_is_metadata_then_features():
    assert ALL_COLUMNS == METADATA_COLUMNS + FEATURE_COLUMNS
