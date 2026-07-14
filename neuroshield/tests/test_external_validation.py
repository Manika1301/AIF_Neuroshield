from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from neuroshield.data.stress_predict_loader import DEFAULT_ROOT
from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.models.artifact import save_model_artifact, train_final_m1
from neuroshield.models.external_validation import (
    render_notes_markdown,
    run_external_validation,
    save_notes,
)

requires_stress_predict = pytest.mark.skipif(
    not (DEFAULT_ROOT / "Raw_data" / "S02").exists(),
    reason="Stress-Predict dataset not present; clone it to run this test",
)


def _separable_kept(n_subjects=3, n_per_class=25, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        for label, shift in ((0, 0.0), (1, 3.0)):
            for _ in range(n_per_class):
                row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                row["subject_id"] = f"S{i}"
                row["m1_label"] = label
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("ext_val_model")
    kept = _separable_kept()
    pipeline = train_final_m1(kept)
    model_path = tmp_dir / "m1.joblib"
    manifest_path = tmp_dir / "m1_manifest.json"
    save_model_artifact(pipeline, kept, model_path=model_path, manifest_path=manifest_path)
    return model_path, manifest_path


@requires_stress_predict
class TestRunExternalValidation:
    def test_runs_and_reports_compatible(self, trained_artifact):
        model_path, manifest_path = trained_artifact
        with patch("neuroshield.models.external_validation.load_model_artifact") as mock_load:
            import neuroshield.models.artifact as artifact_module

            mock_load.return_value = artifact_module.load_model_artifact(model_path, manifest_path)
            result = run_external_validation(subject_ids=["S02", "S03"])

        assert result["compatible"] is True
        assert result["dataset"] == "stress_predict"
        assert result["n_windows_evaluated"] > 0
        assert "balanced_accuracy" in result["overall"]
        assert "macro_f1" in result["overall"]
        assert len(result["overall"]["confusion_matrix"]) == 2

    def test_pilot_subject_without_labels_is_skipped_not_fatal(self, trained_artifact):
        model_path, manifest_path = trained_artifact
        with patch("neuroshield.models.external_validation.load_model_artifact") as mock_load:
            import neuroshield.models.artifact as artifact_module

            mock_load.return_value = artifact_module.load_model_artifact(model_path, manifest_path)
            result = run_external_validation(subject_ids=["S01", "S02"])

        assert result["compatible"] is True
        assert "S01" in result["skipped_subjects"]

    def test_per_subject_metrics_present(self, trained_artifact):
        model_path, manifest_path = trained_artifact
        with patch("neuroshield.models.external_validation.load_model_artifact") as mock_load:
            import neuroshield.models.artifact as artifact_module

            mock_load.return_value = artifact_module.load_model_artifact(model_path, manifest_path)
            result = run_external_validation(subject_ids=["S02", "S03"])

        assert set(result["per_subject"]) == {"S02", "S03"}
        for metrics in result["per_subject"].values():
            assert metrics["n_windows"] > 0


class TestRenderNotesMarkdown:
    def test_compatible_result_mentions_metrics(self):
        result = {
            "compatible": True,
            "model_version": "m1_wesad_features_v1",
            "feature_version": "features-v1",
            "n_subjects_evaluated": 2,
            "n_windows_evaluated": 50,
            "class_balance": {"baseline": 30, "stress": 20},
            "overall": {"balanced_accuracy": 0.6, "macro_f1": 0.55, "confusion_matrix": [[20, 10], [5, 15]]},
            "per_subject": {"S02": {"n_windows": 25, "balanced_accuracy": 0.6}},
            "skipped_subjects": {"S01": "no labels"},
        }
        markdown = render_notes_markdown(result)
        assert "0.600" in markdown
        assert "S02" in markdown
        assert "S01" in markdown
        assert "Channel compatibility" in markdown

    def test_incompatible_result_reports_reason(self):
        result = {
            "compatible": False,
            "incompatibility_reason": "no ACC channel available",
            "skipped_subjects": {},
        }
        markdown = render_notes_markdown(result)
        assert "incompatible" in markdown.lower()
        assert "no ACC channel available" in markdown

    def test_single_class_subject_reports_na(self):
        result = {
            "compatible": True,
            "model_version": "m",
            "feature_version": "f",
            "n_subjects_evaluated": 1,
            "n_windows_evaluated": 10,
            "class_balance": {"baseline": 10, "stress": 0},
            "overall": {"balanced_accuracy": 0.5, "macro_f1": 0.4, "confusion_matrix": [[10, 0], [0, 0]]},
            "per_subject": {"S05": {"n_windows": 10, "balanced_accuracy": None}},
            "skipped_subjects": {},
        }
        markdown = render_notes_markdown(result)
        assert "n/a (single class)" in markdown


def test_save_notes_writes_file(tmp_path):
    result = {
        "compatible": True,
        "model_version": "m",
        "feature_version": "f",
        "n_subjects_evaluated": 1,
        "n_windows_evaluated": 5,
        "class_balance": {"baseline": 5, "stress": 0},
        "overall": {"balanced_accuracy": 0.5, "macro_f1": 0.4, "confusion_matrix": [[5, 0], [0, 0]]},
        "per_subject": {},
        "skipped_subjects": {},
    }
    path = tmp_path / "notes.md"
    save_notes(result, path)
    assert path.exists()
    assert "External validation" in path.read_text()
