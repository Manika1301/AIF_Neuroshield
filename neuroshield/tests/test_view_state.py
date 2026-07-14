from view_state import (
    BACKEND_ERROR,
    DISCONNECTED,
    history_to_dataframe,
    is_abstention_state,
    is_color_state,
    label_and_color,
    latest_values_row,
    quality_row,
)


class TestLabelAndColor:
    def test_known_states_have_labels(self):
        for state in ("waiting", "calibrating", "green", "amber", "red", "motion_paused", "poor_signal", "stale", "error"):
            label, color = label_and_color(state)
            assert label
            assert color

    def test_disconnected_and_backend_error_are_red(self):
        assert label_and_color(DISCONNECTED)[1] == "red"
        assert label_and_color(BACKEND_ERROR)[1] == "red"

    def test_unknown_state_falls_back_gracefully(self):
        label, color = label_and_color("totally_unknown_state")
        assert label == "totally_unknown_state"
        assert color == "gray"

    def test_color_states_map_to_traffic_light_colors(self):
        assert label_and_color("green")[1] == "green"
        assert label_and_color("amber")[1] == "orange"
        assert label_and_color("red")[1] == "red"


class TestQualityRow:
    def test_extracts_expected_keys(self):
        status = {"quality": {"valid_fraction": 0.9, "ppg_quality": 0.8, "motion_dynamic_rms": 0.1, "motion_dynamic_p95": 0.2}}
        row = quality_row(status)
        assert row == {"valid_fraction": 0.9, "ppg_quality": 0.8, "motion_dynamic_rms": 0.1, "motion_dynamic_p95": 0.2}

    def test_missing_quality_key_gives_all_none(self):
        row = quality_row({})
        assert all(v is None for v in row.values())


class TestLatestValuesRow:
    def test_returns_values_dict(self):
        status = {"values": {"hr_mean_bpm": 72.0, "eda_level": 0.3}}
        assert latest_values_row(status) == {"hr_mean_bpm": 72.0, "eda_level": 0.3}

    def test_missing_values_key_gives_empty_dict(self):
        assert latest_values_row({}) == {}


class TestHistoryToDataframe:
    def test_flattens_records_with_values(self):
        records = [
            {"window_start_s": 30.0, "state": "green", "probability": 0.1, "values": {"hr_mean_bpm": 70.0}},
            {"window_start_s": 0.0, "state": "green", "probability": 0.2, "values": {"hr_mean_bpm": 68.0}},
        ]
        df = history_to_dataframe(records)
        assert list(df.columns) >= list(df.columns)  # sanity: no crash
        assert "hr_mean_bpm" in df.columns
        assert df.iloc[0]["window_start_s"] == 0.0  # sorted ascending

    def test_empty_records_gives_empty_dataframe(self):
        df = history_to_dataframe([])
        assert len(df) == 0

    def test_missing_values_key_still_works(self):
        records = [{"window_start_s": 0.0, "state": "waiting", "probability": None}]
        df = history_to_dataframe(records)
        assert len(df) == 1


class TestStateClassification:
    def test_color_states(self):
        assert is_color_state("green")
        assert is_color_state("amber")
        assert is_color_state("red")
        assert not is_color_state("waiting")

    def test_abstention_states(self):
        assert is_abstention_state("motion_paused")
        assert is_abstention_state("poor_signal")
        assert not is_abstention_state("green")
