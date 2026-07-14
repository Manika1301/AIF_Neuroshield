import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.features.personalize import (
    MODEL_FEATURE_COLUMNS,
    PERSONALIZE_BASE,
    PERSONALIZED_COLUMNS,
    add_personalized_features,
)


def _frame(subjects=("A", "B"), n=20, offset_per_subject=None, seed=0) -> pd.DataFrame:
    """Windows for several subjects, each with their own absolute physiological offset."""
    rng = np.random.default_rng(seed)
    offset_per_subject = offset_per_subject or {}
    rows = []
    for s in subjects:
        offset = offset_per_subject.get(s, 0.0)
        for w in range(n):
            row = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
            row["hr_mean_bpm"] = 70 + offset + rng.normal(0, 1)
            row["valid_fraction"] = 1.0
            row["subject_id"] = s
            row["window_start_s"] = w * 30.0
            rows.append(row)
    return pd.DataFrame(rows)


class TestSchema:
    def test_adds_one_personalized_column_per_physiological_feature(self):
        out = add_personalized_features(_frame())
        for col in PERSONALIZED_COLUMNS:
            assert col in out.columns
        assert len(PERSONALIZED_COLUMNS) == len(PERSONALIZE_BASE)

    def test_quality_features_are_not_personalized(self):
        # ppg_quality/valid_fraction describe the recording, not the person.
        assert "ppg_quality_p" not in PERSONALIZED_COLUMNS
        assert "valid_fraction_p" not in PERSONALIZED_COLUMNS

    def test_model_columns_are_raw_plus_personalized(self):
        assert MODEL_FEATURE_COLUMNS == list(FEATURE_COLUMNS) + PERSONALIZED_COLUMNS

    def test_raw_columns_are_left_untouched(self):
        df = _frame()
        out = add_personalized_features(df)
        pd.testing.assert_frame_equal(out[FEATURE_COLUMNS], df[FEATURE_COLUMNS])

    def test_missing_feature_column_raises(self):
        with pytest.raises(ValueError, match="missing feature columns"):
            add_personalized_features(_frame().drop(columns=["hr_mean_bpm"]))


class TestPersonalization:
    def test_removes_between_subject_offsets(self):
        """The point of the whole module: two people who are equally calm look equally calm."""
        df = _frame(subjects=("low", "high"), offset_per_subject={"low": -12.0, "high": +12.0})
        out = add_personalized_features(df)

        raw_gap = abs(
            out.loc[out.subject_id == "low", "hr_mean_bpm"].mean()
            - out.loc[out.subject_id == "high", "hr_mean_bpm"].mean()
        )
        personal_gap = abs(
            out.loc[out.subject_id == "low", "hr_mean_bpm_p"].mean()
            - out.loc[out.subject_id == "high", "hr_mean_bpm_p"].mean()
        )
        assert raw_gap > 20.0
        assert personal_gap < 1.0  # ~24 bpm of pure identity collapses to under one personal SD

    def test_reference_window_is_near_zero_deviation(self):
        out = add_personalized_features(_frame(subjects=("A",)))
        early = out.head(5)["hr_mean_bpm_p"].mean()
        assert abs(early) < 1.5  # a subject's own reference period sits at ~0 by construction

    def test_deviation_tracks_a_real_rise(self):
        df = _frame(subjects=("A",), n=20)
        df.loc[15:, "hr_mean_bpm"] += 15.0  # a genuine within-person arousal rise
        out = add_personalized_features(df)
        assert out.loc[15:, "hr_mean_bpm_p"].mean() > 5.0

    def test_each_subject_gets_its_own_reference(self):
        df = _frame(subjects=("A", "B"), offset_per_subject={"A": 0.0, "B": 30.0})
        out = add_personalized_features(df)
        for s in ("A", "B"):
            assert abs(out.loc[out.subject_id == s, "hr_mean_bpm_p"].mean()) < 1.0


class TestReference:
    def test_low_quality_windows_are_not_used_as_the_reference(self):
        df = _frame(subjects=("A",), n=20)
        df.loc[:4, "valid_fraction"] = 0.1  # a bad start to the recording
        df.loc[:4, "hr_mean_bpm"] = 200.0  # ...with garbage values
        out = add_personalized_features(df)
        # The reference came from the clean windows, so clean windows sit near zero despite the junk.
        assert abs(out.loc[10:, "hr_mean_bpm_p"].mean()) < 2.0

    def test_falls_back_to_all_windows_when_too_few_are_clean(self):
        df = _frame(subjects=("A",), n=20)
        df["valid_fraction"] = 0.1  # nothing meets the quality bar
        out = add_personalized_features(df)
        assert out["hr_mean_bpm_p"].notna().all()  # a noisy reference still beats no reference

    def test_uses_only_the_first_n_windows_so_it_stays_causal(self):
        """A late stress episode must not contaminate the reference (no peeking at the future)."""
        early = add_personalized_features(_frame(subjects=("A",), n=20))
        df = _frame(subjects=("A",), n=20)
        df.loc[15:, "hr_mean_bpm"] += 40.0
        late = add_personalized_features(df)
        assert np.allclose(
            early.loc[:9, "hr_mean_bpm_p"], late.loc[:9, "hr_mean_bpm_p"], equal_nan=True
        )

    def test_requires_subject_column_without_a_profile(self):
        with pytest.raises(ValueError, match="subject_id"):
            add_personalized_features(_frame().drop(columns=["subject_id"]))


class TestProfileMode:
    def test_profile_is_used_verbatim_as_the_reference(self):
        """The live path: the engine's real calibration profile, not a derived one."""
        df = _frame(subjects=("A",), n=10)
        profile = {
            "feature_means": {c: 0.0 for c in FEATURE_COLUMNS} | {"hr_mean_bpm": 70.0},
            "feature_stds": {c: 1.0 for c in FEATURE_COLUMNS},
        }
        out = add_personalized_features(df, profile=profile)
        expected = df["hr_mean_bpm"] - 70.0
        assert np.allclose(out["hr_mean_bpm_p"], expected)

    def test_profile_mode_needs_no_subject_column(self):
        df = _frame(subjects=("A",), n=5).drop(columns=["subject_id"])
        profile = {
            "feature_means": {c: 0.0 for c in FEATURE_COLUMNS},
            "feature_stds": {c: 1.0 for c in FEATURE_COLUMNS},
        }
        out = add_personalized_features(df, profile=profile)
        assert out["hr_mean_bpm_p"].notna().all()
