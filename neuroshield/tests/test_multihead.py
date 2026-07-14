import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS, FEATURE_VERSION
from neuroshield.features.harmonize import harmonize_labels, pool_harmonized
from neuroshield.features.personalize import MODEL_FEATURE_COLUMNS, add_personalized_features
from neuroshield.models.artifact import IncompatibleFeatureVersionError, ModelIntegrityError
from neuroshield.models.multihead import (
    LEVEL_CALM,
    LEVEL_HIGH,
    MODEL_VERSION,
    MultiHeadModel,
    evaluate_head_a,
    evaluate_head_b,
    load_multihead_artifact,
    save_multihead_artifact,
    train_final_multihead,
)


def _wesad_like(n_subjects=6, n_per_class=20, seed=0) -> pd.DataFrame:
    """Synthetic WESAD-shaped rows with all 4 affect classes and a separable stress signal."""
    rng = np.random.default_rng(seed)
    # raw WESAD codes: 1 baseline, 2 stress, 3 amusement, 4 meditation
    rows = []
    for i in range(n_subjects):
        for raw, shift in ((1, 0.0), (2, 3.0), (3, 1.5), (4, -1.0)):
            for _ in range(n_per_class):
                row = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                row["subject_id"] = f"W{i}"
                row["label"] = raw
                row["valid_fraction"] = 1.0
                rows.append(row)
    return add_personalized_features(pd.DataFrame(rows))


def _sp_like(n_subjects=6, n_per_class=20, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        for raw, shift in ((0, 0.0), (1, 3.0)):
            for _ in range(n_per_class):
                row = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                row["subject_id"] = f"P{i}"
                row["label"] = raw
                row["valid_fraction"] = 1.0
                rows.append(row)
    return add_personalized_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def pooled():
    w, _ = harmonize_labels(_wesad_like(), "wesad")
    sp, _ = harmonize_labels(_sp_like(), "stress_predict")
    return pool_harmonized([w, sp])


@pytest.fixture(scope="module")
def model(pooled):
    return train_final_multihead(pooled, random_state=0)


class TestEvaluation:
    def test_head_a_beats_dummy(self, pooled):
        result = evaluate_head_a(pooled, random_state=0)
        assert result["model"]["balanced_accuracy"] > result["dummy"]["balanced_accuracy"]
        assert result["dummy"]["balanced_accuracy"] == pytest.approx(0.5, abs=1e-6)
        assert set(result["datasets"]) == {"wesad"}  # Head A trains on WESAD only now

    def test_head_b_beats_dummy_and_is_four_class(self, pooled):
        result = evaluate_head_b(pooled, random_state=0)
        assert result["model"]["balanced_accuracy"] > result["dummy"]["balanced_accuracy"]
        assert np.array(result["model"]["confusion_matrix"]).shape == (4, 4)

    def test_head_a_requires_multiple_groups(self):
        w, _ = harmonize_labels(_wesad_like(n_subjects=1), "wesad")
        with pytest.raises(ValueError, match="2 groups"):
            evaluate_head_a(pool_harmonized([w]))


class TestPredict:
    def test_predict_output_shape_and_columns(self, model, pooled):
        sample = pooled.head(10)
        out = model.predict(sample)
        assert len(out) == 10
        for col in ("stress_prob", "stress_index", "level", "affect_state", "affect_confidence"):
            assert col in out.columns

    def test_stress_index_in_0_100(self, model, pooled):
        out = model.predict(pooled)
        assert out["stress_index"].between(0, 100).all()

    def test_index_increases_with_stress_signal(self, model):
        """Calibration monotonicity: a clearly high-stress row scores higher than a calm row."""
        base = {c: 0.0 for c in MODEL_FEATURE_COLUMNS}
        calm = dict(base, hr_mean_bpm=65.0, eda_level=0.0, hr_mean_bpm_p=0.0, eda_level_p=0.0)
        stressed = dict(base, hr_mean_bpm=95.0, eda_level=1.0, hr_mean_bpm_p=6.0, eda_level_p=6.0)
        out = model.predict(pd.DataFrame([calm, stressed]))
        assert out["stress_index"].iloc[1] > out["stress_index"].iloc[0]

    def test_levels_map_to_thresholds(self, model):
        base = {c: 0.0 for c in MODEL_FEATURE_COLUMNS}
        calm = dict(base, hr_mean_bpm=63.0, eda_level=-0.2, hr_mean_bpm_p=-1.0, eda_level_p=-1.0)
        stressed = dict(base, hr_mean_bpm=100.0, eda_level=1.2, hr_mean_bpm_p=8.0, eda_level_p=8.0)
        out = model.predict(pd.DataFrame([calm, stressed]))
        assert out["level"].iloc[0] == LEVEL_CALM
        assert out["level"].iloc[1] == LEVEL_HIGH

    def test_affect_state_is_a_known_class(self, model, pooled):
        out = model.predict(pooled.head(20))
        assert out["affect_state"].isin(["baseline", "stress", "amusement", "meditation"]).all()

    def test_missing_column_raises(self, model, pooled):
        with pytest.raises(ValueError, match="missing required columns"):
            model.predict(pooled.drop(columns=["eda_level"]))


class TestArtifact:
    def test_save_load_roundtrip(self, tmp_path, model, pooled):
        mp = tmp_path / "m2.joblib"
        manp = tmp_path / "m2_manifest.json"
        metp = tmp_path / "m2_metrics.json"
        manifest = save_multihead_artifact(model, pooled, metrics_path=metp, model_path=mp, manifest_path=manp)

        assert manifest["model_version"] == MODEL_VERSION
        assert manifest["feature_version"] == FEATURE_VERSION
        assert manifest["heads"]["head_a"]["training_datasets"] == ["wesad"]
        assert manifest["heads"]["head_b"]["training_datasets"] == ["wesad"]

        loaded, loaded_manifest = load_multihead_artifact(model_path=mp, manifest_path=manp)
        assert isinstance(loaded, MultiHeadModel)
        a = model.predict(pooled.head(5))
        b = loaded.predict(pooled.head(5))
        assert np.allclose(a["stress_prob"], b["stress_prob"])

    def test_feature_version_mismatch_refuses(self, tmp_path, model, pooled):
        import json

        mp = tmp_path / "m2.joblib"
        manp = tmp_path / "m2_manifest.json"
        save_multihead_artifact(model, pooled, metrics_path=tmp_path / "m.json", model_path=mp, manifest_path=manp)
        manifest = json.loads(manp.read_text())
        manifest["feature_version"] = "features-v0"
        manp.write_text(json.dumps(manifest))
        with pytest.raises(IncompatibleFeatureVersionError):
            load_multihead_artifact(model_path=mp, manifest_path=manp)

    def test_corrupted_model_refuses(self, tmp_path, model, pooled):
        mp = tmp_path / "m2.joblib"
        manp = tmp_path / "m2_manifest.json"
        save_multihead_artifact(model, pooled, metrics_path=tmp_path / "m.json", model_path=mp, manifest_path=manp)
        mp.write_bytes(b"corrupted")
        with pytest.raises(ModelIntegrityError):
            load_multihead_artifact(model_path=mp, manifest_path=manp)


def test_does_not_touch_m1_artifact_paths():
    from neuroshield.models.artifact import DEFAULT_MODEL_PATH as M1_PATH
    from neuroshield.models.multihead import DEFAULT_MODEL_PATH as M2_PATH

    assert M1_PATH != M2_PATH  # m1 is never overwritten by the multihead build
