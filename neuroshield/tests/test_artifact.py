import hashlib

import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS, FEATURE_VERSION
from neuroshield.models.artifact import (
    IncompatibleFeatureVersionError,
    MissingFeatureColumnsError,
    ModelIntegrityError,
    load_model_artifact,
    predict_proba_stress,
    save_model_artifact,
    train_final_m1,
)


def _synthetic_kept(n_subjects: int = 4, n_per_class: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        for label in (0, 1):
            shift = 0.0 if label == 0 else 3.0
            for _ in range(n_per_class):
                row = {col: rng.normal(0, 1) for col in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift + rng.normal(0, 0.3)
                row["subject_id"] = f"S{i}"
                row["m1_label"] = label
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def kept():
    return _synthetic_kept()


@pytest.fixture(scope="module")
def pipeline(kept):
    return train_final_m1(kept)


@pytest.fixture
def saved_artifact(tmp_path, pipeline, kept):
    model_path = tmp_path / "m1.joblib"
    manifest_path = tmp_path / "m1_manifest.json"
    metrics_path = tmp_path / "m1_loso_summary.json"
    manifest = save_model_artifact(
        pipeline,
        kept,
        metrics_path=metrics_path,
        model_path=model_path,
        manifest_path=manifest_path,
    )
    return {
        "manifest": manifest,
        "model_path": model_path,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
    }


class TestManifestContents:
    def test_required_fields_present(self, saved_artifact):
        manifest = saved_artifact["manifest"]
        for key in (
            "model_version",
            "feature_version",
            "feature_columns",
            "training_dataset",
            "code_commit",
            "metrics_path",
            "threshold_policy",
            "checksum_sha256",
        ):
            assert key in manifest

    def test_feature_column_order_preserved(self, saved_artifact):
        assert saved_artifact["manifest"]["feature_columns"] == FEATURE_COLUMNS

    def test_feature_version_matches_extractor(self, saved_artifact):
        assert saved_artifact["manifest"]["feature_version"] == FEATURE_VERSION

    def test_checksum_matches_file_on_disk(self, saved_artifact):
        digest = hashlib.sha256(saved_artifact["model_path"].read_bytes()).hexdigest()
        assert digest == saved_artifact["manifest"]["checksum_sha256"]

    def test_threshold_policy_has_green_and_amber_bounds(self, saved_artifact):
        policy = saved_artifact["manifest"]["threshold_policy"]
        assert "green_max" in policy
        assert "amber_max" in policy
        assert policy["green_max"] < policy["amber_max"]


class TestLoadRoundtrip:
    def test_load_returns_working_pipeline(self, saved_artifact, kept):
        loaded_pipeline, manifest = load_model_artifact(
            model_path=saved_artifact["model_path"], manifest_path=saved_artifact["manifest_path"]
        )
        X = kept[manifest["feature_columns"]].to_numpy(dtype=float)
        probs = loaded_pipeline.predict_proba(X)[:, 1]
        assert probs.shape[0] == len(kept)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_incompatible_feature_version_refuses_to_load(self, saved_artifact):
        import json

        manifest_path = saved_artifact["manifest_path"]
        manifest = json.loads(manifest_path.read_text())
        manifest["feature_version"] = "features-v0"
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(IncompatibleFeatureVersionError):
            load_model_artifact(
                model_path=saved_artifact["model_path"], manifest_path=manifest_path
            )

    def test_corrupted_model_file_refuses_to_load(self, saved_artifact):
        saved_artifact["model_path"].write_bytes(b"not a real joblib file")
        with pytest.raises(ModelIntegrityError):
            load_model_artifact(
                model_path=saved_artifact["model_path"], manifest_path=saved_artifact["manifest_path"]
            )

    def test_missing_manifest_raises_file_not_found(self, tmp_path, saved_artifact):
        with pytest.raises(FileNotFoundError):
            load_model_artifact(
                model_path=saved_artifact["model_path"], manifest_path=tmp_path / "nope.json"
            )


class TestPredictProbaStress:
    def test_reorders_shuffled_columns_correctly(self, saved_artifact, kept):
        loaded_pipeline, manifest = load_model_artifact(
            model_path=saved_artifact["model_path"], manifest_path=saved_artifact["manifest_path"]
        )
        shuffled_cols = list(reversed(FEATURE_COLUMNS)) + ["subject_id", "m1_label"]
        shuffled = kept[shuffled_cols]

        probs_via_helper = predict_proba_stress(loaded_pipeline, manifest, shuffled)
        X_correct_order = kept[FEATURE_COLUMNS].to_numpy(dtype=float)
        probs_direct = loaded_pipeline.predict_proba(X_correct_order)[:, 1]

        assert np.allclose(probs_via_helper, probs_direct)

    def test_missing_required_column_raises(self, saved_artifact, kept):
        loaded_pipeline, manifest = load_model_artifact(
            model_path=saved_artifact["model_path"], manifest_path=saved_artifact["manifest_path"]
        )
        incomplete = kept.drop(columns=["eda_level"])
        with pytest.raises(MissingFeatureColumnsError):
            predict_proba_stress(loaded_pipeline, manifest, incomplete)
