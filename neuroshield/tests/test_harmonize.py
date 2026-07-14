import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.features.harmonize import (
    AFFECT_CLASSES,
    HEAD_A_BASELINE,
    HEAD_A_STRESS,
    harmonize_labels,
    participant_group,
    pool_harmonized,
    training_view,
)


def _features(dataset_rows):
    """dataset_rows: list of (subject_id, raw_label, valid_fraction)."""
    rows = []
    for sid, label, vf in dataset_rows:
        row = {col: 0.0 for col in FEATURE_COLUMNS}
        row["subject_id"] = sid
        row["label"] = label
        row["valid_fraction"] = vf
        rows.append(row)
    return pd.DataFrame(rows)


class TestParticipantGroup:
    def test_regular_dataset_uses_subject_as_group(self):
        assert participant_group("wesad", "S2") == "wesad:S2"

    def test_nurse_session_collapses_to_participant(self):
        assert participant_group("nurse_stress", "E4_1587206108") == "nurse_stress:E4"
        assert participant_group("nurse_stress", "E4_1599999999") == "nurse_stress:E4"


class TestWesadHarmonization:
    def test_head_a_maps_baseline_and_stress_only(self):
        # WESAD raw: 1=baseline,2=stress,3=amusement,4=meditation,0=transient
        feats = _features([("S2", 1, 1.0), ("S2", 2, 1.0), ("S2", 3, 1.0), ("S2", 4, 1.0), ("S2", 0, 1.0)])
        df, _ = harmonize_labels(feats, "wesad")
        assert df["head_a_label"].tolist()[:2] == [HEAD_A_BASELINE, HEAD_A_STRESS]
        assert df["head_a_label"].iloc[2:].isna().all()  # amusement/meditation/transient excluded from head A

    def test_head_b_maps_all_four_affect_classes(self):
        feats = _features([("S2", 1, 1.0), ("S2", 2, 1.0), ("S2", 3, 1.0), ("S2", 4, 1.0), ("S2", 0, 1.0)])
        df, _ = harmonize_labels(feats, "wesad")
        assert df["head_b_label"].tolist()[:4] == [0, 1, 2, 3]
        assert np.isnan(df["head_b_label"].iloc[4])  # transient has no affect class

    def test_split_is_train_pool(self):
        df, counts = harmonize_labels(_features([("S2", 1, 1.0)]), "wesad")
        assert (df["split"] == "train_pool").all()
        assert counts["split"] == "train_pool"


class TestBinaryDatasetHarmonization:
    def test_stress_predict_binary_maps_to_head_a(self):
        feats = _features([("S01", 0, 1.0), ("S01", 1, 1.0), ("S01", -1, 1.0)])
        df, _ = harmonize_labels(feats, "stress_predict")
        assert df["head_a_label"].tolist()[:2] == [HEAD_A_BASELINE, HEAD_A_STRESS]
        assert np.isnan(df["head_a_label"].iloc[2])  # -1 unlabeled

    def test_binary_datasets_have_no_head_b_labels(self):
        df, _ = harmonize_labels(_features([("S01", 1, 1.0)]), "stress_predict")
        assert df["head_b_label"].isna().all()

    def test_nurse_is_heldout_split(self):
        df, counts = harmonize_labels(_features([("E4_100", 1, 1.0)]), "nurse_stress")
        assert (df["split"] == "heldout").all()
        assert counts["split"] == "heldout"


class TestQualityGate:
    def test_low_valid_fraction_is_ineligible_but_keeps_label(self):
        feats = _features([("S2", 2, 0.4)])  # stress label but poor coverage
        df, counts = harmonize_labels(feats, "wesad", min_valid_fraction=0.9)
        assert df["head_a_label"].iloc[0] == HEAD_A_STRESS  # label preserved
        assert not df["eligible_head_a"].iloc[0]  # but not eligible
        assert counts["head_a"]["excluded_low_quality"] == 1


class TestCounts:
    def test_counts_are_accurate(self):
        feats = _features(
            [("S2", 1, 1.0), ("S2", 2, 1.0), ("S2", 3, 1.0), ("S2", 2, 0.1), ("S2", 0, 1.0)]
        )
        _, counts = harmonize_labels(feats, "wesad", min_valid_fraction=0.9)
        # head A eligible: baseline(1) + stress(1) = 2; the 0.1-coverage stress is low-quality
        assert counts["head_a"]["eligible"] == 2
        assert counts["head_a"]["baseline"] == 1
        assert counts["head_a"]["stress"] == 1
        assert counts["head_a"]["excluded_low_quality"] == 1
        assert counts["head_a"]["excluded_no_label"] == 2  # amusement(3) + transient(0)
        # head B eligible: baseline, stress, amusement (3) = 3 clean ones
        assert counts["head_b"]["eligible"] == 3
        assert counts["head_b"]["per_class"] == {"baseline": 1, "stress": 1, "amusement": 1, "meditation": 0}


class TestPoolingAndTrainingView:
    def _pooled(self):
        w, _ = harmonize_labels(_features([("S2", 1, 1.0), ("S2", 2, 1.0), ("S2", 3, 1.0)]), "wesad")
        sp, _ = harmonize_labels(_features([("S01", 0, 1.0), ("S01", 1, 1.0)]), "stress_predict")
        n, _ = harmonize_labels(_features([("E4_1", 1, 1.0), ("E4_1", 0, 1.0)]), "nurse_stress")
        return pool_harmonized([w, sp, n])

    def test_pool_concatenates_all(self):
        pooled = self._pooled()
        assert len(pooled) == 7
        assert set(pooled["dataset"]) == {"wesad", "stress_predict", "nurse_stress"}

    def test_head_a_training_view_is_wesad_only(self):
        # Head A trains on WESAD only; Stress-Predict and Nurse are held out for validation.
        pooled = self._pooled()
        train = training_view(pooled, "head_a")
        assert set(train["dataset"]) == {"wesad"}
        assert {"stress_predict", "nurse_stress"}.isdisjoint(set(train["dataset"]))

    def test_head_b_training_view_is_wesad_only(self):
        pooled = self._pooled()
        train = training_view(pooled, "head_b")
        assert set(train["dataset"]) == {"wesad"}

    def test_no_group_appears_in_both_train_pool_and_heldout(self):
        pooled = self._pooled()
        train_groups = set(pooled.loc[pooled["split"] == "train_pool", "group"])
        heldout_groups = set(pooled.loc[pooled["split"] == "heldout", "group"])
        assert train_groups.isdisjoint(heldout_groups)


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        harmonize_labels(_features([("X", 1, 1.0)]), "made_up")


def test_affect_classes_are_distinct_and_four():
    assert set(AFFECT_CLASSES.values()) == {0, 1, 2, 3}
