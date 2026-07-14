import json

import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.models.train_m1 import save_results, train_and_evaluate_m1


def _synthetic_kept(n_subjects: int = 5, n_per_class: int = 30, seed: int = 0) -> pd.DataFrame:
    """Windows with a subject-independent, clearly separable baseline-vs-stress signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        subject_id = f"S{i}"
        for label in (0, 1):
            shift = 0.0 if label == 0 else 3.0  # well-separated in the informative features
            for _ in range(n_per_class):
                row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift + rng.normal(0, 0.3)
                row["subject_id"] = subject_id
                row["m1_label"] = label
                rows.append(row)
    df = pd.DataFrame(rows)
    # Sprinkle a few NaNs to exercise the imputer, without touching the separable columns.
    df.loc[df.sample(frac=0.05, random_state=seed).index, "motion_dynamic_p95"] = np.nan
    return df


@pytest.fixture(scope="module")
def kept():
    return _synthetic_kept()


@pytest.fixture(scope="module")
def result(kept):
    return train_and_evaluate_m1(kept, random_state=0)


class TestLOSOStructure:
    def test_every_window_predicted_exactly_once(self, kept, result):
        predictions = result["predictions"]
        assert len(predictions) == len(kept)

    def test_every_subject_appears_as_its_own_held_out_fold(self, kept, result):
        predictions = result["predictions"]
        assert set(predictions["subject_id"]) == set(kept["subject_id"])
        for subject_id, group in kept.groupby("subject_id"):
            assert (predictions["subject_id"] == subject_id).sum() == len(group)

    def test_raises_on_single_subject(self):
        single = _synthetic_kept(n_subjects=1)
        with pytest.raises(ValueError, match="at least 2 subjects"):
            train_and_evaluate_m1(single)


class TestMetrics:
    def test_overall_metrics_in_valid_range(self, result):
        for model_key in ("model", "dummy"):
            m = result["summary"]["overall"][model_key]
            assert 0.0 <= m["balanced_accuracy"] <= 1.0
            assert 0.0 <= m["macro_f1"] <= 1.0

    def test_confusion_matrix_shape_and_total(self, kept, result):
        cm = np.array(result["summary"]["overall"]["model"]["confusion_matrix"])
        assert cm.shape == (2, 2)
        assert cm.sum() == len(kept)

    def test_per_subject_metrics_cover_all_subjects(self, kept, result):
        per_subject = result["summary"]["per_subject"]
        assert set(per_subject.keys()) == set(kept["subject_id"].unique())
        for metrics in per_subject.values():
            assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

    def test_class_balance_reported(self, kept, result):
        balance = result["summary"]["class_balance"]
        assert balance["baseline"] == int((kept["m1_label"] == 0).sum())
        assert balance["stress"] == int((kept["m1_label"] == 1).sum())

    def test_model_beats_dummy_on_separable_signal(self, result):
        model_ba = result["summary"]["overall"]["model"]["balanced_accuracy"]
        dummy_ba = result["summary"]["overall"]["dummy"]["balanced_accuracy"]
        assert model_ba > dummy_ba
        assert dummy_ba == pytest.approx(0.5, abs=1e-6)  # most_frequent dummy is chance-level

    def test_predictions_have_expected_columns(self, result):
        expected = {"subject_id", "y_true", "y_pred_model", "y_prob_model", "y_pred_dummy"}
        assert expected.issubset(result["predictions"].columns)


def test_save_results_writes_predictions_and_summary(tmp_path, result):
    pred_path = tmp_path / "m1_loso_predictions.csv"
    summary_path = tmp_path / "m1_loso_summary.json"
    save_results(result, predictions_path=pred_path, summary_path=summary_path)

    assert pred_path.exists()
    reloaded_preds = pd.read_csv(pred_path)
    assert len(reloaded_preds) == len(result["predictions"])

    assert summary_path.exists()
    with open(summary_path) as f:
        reloaded_summary = json.load(f)
    assert reloaded_summary["model_type"] == "logistic_regression"
    assert reloaded_summary["n_subjects"] == result["summary"]["n_subjects"]
