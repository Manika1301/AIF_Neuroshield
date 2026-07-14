import json

import numpy as np
import pandas as pd
import pytest

from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.features.harmonize import harmonize_labels, pool_harmonized
from neuroshield.features.personalize import add_personalized_features
from neuroshield.models.multihead import train_final_multihead
from neuroshield.models.scoreboard import (
    build_scoreboard,
    heldout_predictions,
    loso_predictions_by_dataset,
    render_scoreboard_markdown,
    save_scoreboard,
)


def _binary_rows(prefix, n_subjects, n_per_class, seed, dataset_label_col):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        for raw, shift in ((dataset_label_col[0], 0.0), (dataset_label_col[1], 3.0)):
            for _ in range(n_per_class):
                row = {c: rng.normal(0, 1) for c in FEATURE_COLUMNS}
                row["hr_mean_bpm"] = 65 + shift * 5 + rng.normal(0, 2)
                row["eda_level"] = shift * 0.3 + rng.normal(0, 0.1)
                row["subject_id"] = f"{prefix}{i}"
                row["label"] = raw
                row["valid_fraction"] = 1.0
                rows.append(row)
    return add_personalized_features(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def pooled_train():
    w, _ = harmonize_labels(_binary_rows("W", 5, 20, 0, (1, 2)), "wesad")
    sp, _ = harmonize_labels(_binary_rows("P", 5, 20, 1, (0, 1)), "stress_predict")
    return pool_harmonized([w, sp])


@pytest.fixture(scope="module")
def heldout_pooled():
    # Stress-Predict and Nurse are both held out now (WESAD-only training).
    sp, _ = harmonize_labels(_binary_rows("Q", 4, 15, 3, (0, 1)), "stress_predict")
    n, _ = harmonize_labels(_binary_rows("E4_", 4, 15, 2, (0, 1)), "nurse_stress")
    return pool_harmonized([sp, n])


@pytest.fixture(scope="module")
def model(pooled_train):
    return train_final_multihead(pooled_train, random_state=0)


class TestLosoByDataset:
    def test_buckets_predictions_by_dataset(self, pooled_train):
        # Only WESAD is in the training pool now, so LOSO produces only WESAD predictions.
        preds = loso_predictions_by_dataset(pooled_train, random_state=0)
        assert set(preds["dataset"]) == {"wesad"}
        assert {"y_true", "y_pred"}.issubset(preds.columns)

    def test_prediction_count_matches_eligible_windows(self, pooled_train):
        from neuroshield.features.harmonize import training_view

        preds = loso_predictions_by_dataset(pooled_train, random_state=0)
        assert len(preds) == len(training_view(pooled_train, "head_a"))


class TestHeldout:
    def test_predicts_only_eligible_rows(self, model, heldout_pooled):
        held = heldout_predictions(model, heldout_pooled)
        assert set(held["dataset"]) == {"stress_predict", "nurse_stress"}
        assert len(held) == int(heldout_pooled["eligible_head_a"].sum())

    def test_empty_when_no_eligible(self, model):
        empty = pd.DataFrame({"eligible_head_a": [], "dataset": [], "head_a_label": []})
        held = heldout_predictions(model, empty)
        assert held.empty


class TestBuildScoreboard:
    def test_train_pool_rows_use_loso(self, pooled_train):
        sb = build_scoreboard(pooled_train, random_state=0)
        assert set(sb["per_dataset"]) == {"wesad"}
        for m in sb["per_dataset"].values():
            assert "grouped-LOSO" in m["evaluation"]
            assert 0.0 <= m["balanced_accuracy"] <= 1.0

    def test_heldout_rows_added_when_model_and_heldout_given(self, pooled_train, model, heldout_pooled):
        sb = build_scoreboard(pooled_train, model=model, heldout_pooled=heldout_pooled, random_state=0)
        for ds in ("stress_predict", "nurse_stress"):
            assert ds in sb["per_dataset"]
            assert "held-out" in sb["per_dataset"][ds]["evaluation"]
        assert "grouped-LOSO" in sb["per_dataset"]["wesad"]["evaluation"]

    def test_separable_signal_scores_above_chance(self, pooled_train):
        sb = build_scoreboard(pooled_train, random_state=0)
        for m in sb["per_dataset"].values():
            assert m["balanced_accuracy"] > 0.6  # clearly separable synthetic signal

    def test_note_mentions_honesty_caveats(self, pooled_train):
        sb = build_scoreboard(pooled_train, random_state=0)
        assert "held out" in sb["note"].lower()
        assert "leave-one-subject-out" in sb["note"].lower()


class TestRenderAndSave:
    def test_markdown_has_a_row_per_dataset(self, pooled_train, model, heldout_pooled):
        sb = build_scoreboard(pooled_train, model=model, heldout_pooled=heldout_pooled, random_state=0)
        md = render_scoreboard_markdown(sb)
        assert "wesad" in md and "stress_predict" in md and "nurse_stress" in md
        assert "Balanced acc." in md

    def test_save_writes_json_and_md(self, tmp_path, pooled_train):
        sb = build_scoreboard(pooled_train, random_state=0)
        jp = tmp_path / "scoreboard.json"
        mp = tmp_path / "scoreboard.md"
        save_scoreboard(sb, json_path=jp, md_path=mp)
        assert jp.exists() and mp.exists()
        reloaded = json.loads(jp.read_text())
        assert reloaded["head"] == "head_a"


def test_loso_requires_multiple_groups():
    w, _ = harmonize_labels(_binary_rows("W", 1, 20, 0, (1, 2)), "wesad")
    with pytest.raises(ValueError, match="2 groups"):
        loso_predictions_by_dataset(pool_harmonized([w]))
