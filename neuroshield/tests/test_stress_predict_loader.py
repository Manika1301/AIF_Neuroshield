from pathlib import Path

import numpy as np
import pytest

from neuroshield.data.bundle import stress_predict_subject_to_bundle
from neuroshield.data.stress_predict_loader import DEFAULT_ROOT, load_stress_predict_subject

S02_DIR = DEFAULT_ROOT / "Raw_data" / "S02"
requires_stress_predict = pytest.mark.skipif(
    not S02_DIR.exists(),
    reason=f"Stress-Predict subject not found at {S02_DIR}; clone the dataset to run this test",
)


@requires_stress_predict
class TestStressPredictSubject:
    def test_subject_id_matches(self):
        subject = load_stress_predict_subject("S02")
        assert subject.subject_id == "S02"

    def test_expected_channel_names(self):
        subject = load_stress_predict_subject("S02")
        assert set(subject.sample_rates_hz) == {"BVP", "EDA", "TEMP", "ACC"}

    def test_channel_arrays_non_empty(self):
        subject = load_stress_predict_subject("S02")
        assert subject.bvp.shape[0] > 0
        assert subject.eda.shape[0] > 0
        assert subject.temp.shape[0] > 0
        assert subject.acc.shape[0] > 0

    def test_acc_is_three_axis(self):
        subject = load_stress_predict_subject("S02")
        assert subject.acc.shape[1] == 3

    def test_declared_sample_rates_match_e4_defaults(self):
        subject = load_stress_predict_subject("S02")
        assert subject.sample_rates_hz == {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}

    def test_labels_are_binary_or_unlabeled(self):
        subject = load_stress_predict_subject("S02")
        for arr in subject.labels.values():
            assert set(np.unique(arr).tolist()) <= {-1, 0, 1}

    def test_labels_contain_both_classes_for_a_labelled_subject(self):
        subject = load_stress_predict_subject("S02")
        values = set(subject.labels["EDA"].tolist())
        assert 0 in values
        assert 1 in values

    def test_converts_to_signal_bundle(self):
        subject = load_stress_predict_subject("S02")
        bundle = stress_predict_subject_to_bundle(subject)
        assert bundle.dataset == "stress_predict"
        assert bundle.subject_id == "S02"
        assert bundle.channel_names == ["ACC", "BVP", "EDA", "TEMP"]
        for name in bundle.channel_names:
            assert np.all(np.diff(bundle.time_s[name]) >= 0)

    def test_unlabelled_pilot_subject_has_no_known_labels(self):
        # S01 exists as a raw folder but was excluded from the processed label table.
        subject = load_stress_predict_subject("S01")
        for arr in subject.labels.values():
            assert set(np.unique(arr).tolist()) == {-1}

    def test_missing_subject_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_stress_predict_subject("S99", root=Path("data/external/stress_predict"))
