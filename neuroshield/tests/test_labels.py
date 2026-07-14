import pandas as pd
import pytest

from neuroshield.features.labels import label_m1_binary, save_label_counts


def _windows(rows):
    """rows: list of (subject_id, raw_label, valid_fraction) tuples."""
    return pd.DataFrame(
        [
            {"subject_id": sid, "label": label, "valid_fraction": vf, "hr_mean_bpm": 70.0}
            for sid, label, vf in rows
        ]
    )


@pytest.fixture
def mixed_windows():
    return _windows(
        [
            # S2: 3 clean baseline, 2 clean stress
            ("S2", 1, 1.0),
            ("S2", 1, 0.95),
            ("S2", 1, 1.0),
            ("S2", 2, 1.0),
            ("S2", 2, 0.92),
            # S2: excluded non-binary conditions
            ("S2", 3, 1.0),  # amusement
            ("S2", 4, 1.0),  # meditation
            ("S2", 0, 1.0),  # transient/undefined
            ("S2", 5, 1.0),  # unused/ignore code
            # S2: a baseline and a stress window with poor coverage -> dropped separately
            ("S2", 1, 0.4),
            ("S2", 2, 0.1),
            # S3: 1 baseline, 1 stress, both clean
            ("S3", 1, 1.0),
            ("S3", 2, 1.0),
        ]
    )


class TestLabelM1Binary:
    def test_kept_rows_have_exactly_one_binary_label(self, mixed_windows):
        kept, _ = label_m1_binary(mixed_windows)
        assert kept["m1_label"].isin([0, 1]).all()
        assert kept["m1_label"].notna().all()

    def test_baseline_maps_to_zero_stress_to_one(self, mixed_windows):
        kept, _ = label_m1_binary(mixed_windows)
        assert set(kept.loc[kept["label"] == 1, "m1_label"]) == {0}
        assert set(kept.loc[kept["label"] == 2, "m1_label"]) == {1}

    def test_subject_ids_preserved(self, mixed_windows):
        kept, counts = label_m1_binary(mixed_windows)
        assert set(kept["subject_id"]) == {"S2", "S3"}
        assert "S2" in set(counts["subject_id"])
        assert "S3" in set(counts["subject_id"])

    def test_correct_kept_counts(self, mixed_windows):
        kept, _ = label_m1_binary(mixed_windows)
        # S2: 3 baseline + 2 stress clean; S3: 1 baseline + 1 stress clean
        assert len(kept) == 7
        assert (kept["m1_label"] == 0).sum() == 4  # 3 (S2) + 1 (S3)
        assert (kept["m1_label"] == 1).sum() == 3  # 2 (S2) + 1 (S3)

    def test_excluded_windows_are_counted_by_reason(self, mixed_windows):
        _, counts = label_m1_binary(mixed_windows)
        all_counts = counts[counts["subject_id"] == "ALL"].set_index("category")["count"]
        assert all_counts["excluded_low_valid_fraction"] == 2  # the poor-coverage baseline+stress
        assert all_counts["non_binary_label:amusement"] == 1
        assert all_counts["non_binary_label:meditation"] == 1
        assert all_counts["non_binary_label:transient"] == 1
        assert all_counts["non_binary_label:ignore_5"] == 1

    def test_class_balance_visible_before_training(self, mixed_windows):
        _, counts = label_m1_binary(mixed_windows)
        all_counts = counts[counts["subject_id"] == "ALL"].set_index("category")["count"]
        assert all_counts["baseline"] == 4
        assert all_counts["stress"] == 3

    def test_per_subject_counts_sum_to_all(self, mixed_windows):
        _, counts = label_m1_binary(mixed_windows)
        per_subject_total = counts.loc[counts["subject_id"] != "ALL", "count"].sum()
        all_total = counts.loc[counts["subject_id"] == "ALL", "count"].sum()
        assert per_subject_total == all_total == len(mixed_windows)

    def test_custom_valid_fraction_threshold(self, mixed_windows):
        # Loosen the threshold so the 0.4-coverage baseline window is now kept.
        kept, _ = label_m1_binary(mixed_windows, min_valid_fraction=0.3)
        assert len(kept) == 8


def test_save_label_counts_writes_csv(tmp_path, mixed_windows):
    _, counts = label_m1_binary(mixed_windows)
    out_path = tmp_path / "wesad_label_counts.csv"
    save_label_counts(counts, path=out_path)
    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert set(reloaded.columns) == {"subject_id", "category", "count"}
    assert len(reloaded) == len(counts)
