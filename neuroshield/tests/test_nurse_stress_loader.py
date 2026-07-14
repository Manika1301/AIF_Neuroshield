import numpy as np
import pandas as pd
import pytest

from neuroshield.data.bundle import nurse_stress_session_to_bundle
from neuroshield.data.nurse_stress_loader import (
    DEFAULT_ROOT,
    STRESS_ZIP_NAME,
    UNLABELED,
    label_session_samples,
    list_participant_sessions,
    list_participants,
    load_nurse_stress_session,
    load_survey_events,
)

DATA_PRESENT = (DEFAULT_ROOT / STRESS_ZIP_NAME).exists()
requires_nurse = pytest.mark.skipif(not DATA_PRESENT, reason="Nurse Stress dataset zip not present")


class TestLabelSessionSamples:
    """Interval-overlap labelling logic, tested without the real dataset."""

    def _events(self):
        # one stress (1) event 100-200s and one baseline (0) event 300-400s for participant 'X'
        return pd.DataFrame(
            {
                "participant_id": ["X", "X", "Y"],
                "start_utc": pd.to_datetime([100, 300, 500], unit="s", utc=True),
                "end_utc": pd.to_datetime([200, 400, 600], unit="s", utc=True),
                "binary_label": [1.0, 0.0, 1.0],
            }
        )

    def test_samples_inside_stress_interval_get_label_1(self):
        times = np.array([50, 150, 250, 350, 450], dtype=float)
        labels = label_session_samples(times, "X", self._events())
        assert labels.tolist() == [UNLABELED, 1, UNLABELED, 0, UNLABELED]

    def test_other_participants_events_are_ignored(self):
        times = np.array([550], dtype=float)  # inside Y's interval, but we ask for X
        labels = label_session_samples(times, "X", self._events())
        assert labels.tolist() == [UNLABELED]

    def test_no_events_yields_all_unlabeled(self):
        empty = pd.DataFrame(columns=["participant_id", "start_utc", "end_utc", "binary_label"])
        labels = label_session_samples(np.array([1.0, 2.0]), "X", empty)
        assert set(labels.tolist()) == {UNLABELED}

    def test_boundaries_are_inclusive(self):
        times = np.array([100, 200], dtype=float)  # exactly the interval edges
        labels = label_session_samples(times, "X", self._events())
        assert labels.tolist() == [1, 1]


@requires_nurse
class TestRealDataStructure:
    @staticmethod
    @pytest.fixture(scope="class")
    def events():
        return load_survey_events()

    def test_fifteen_participants(self):
        assert len(list_participants()) == 15

    def test_survey_has_binary_and_excluded_labels(self, events):
        # 179 level-2 (stress) + 46 level-0 (baseline) are labelled; level 1 and 'na' excluded.
        assert events["binary_label"].notna().sum() == 225
        assert (events["binary_label"] == 1).sum() == 179
        assert (events["binary_label"] == 0).sum() == 46

    def test_survey_timestamps_are_utc(self, events):
        labelled = events[events["binary_label"].notna()]
        assert str(labelled["start_utc"].dt.tz) == "UTC"

    def test_session_loads_with_expected_channels(self, events):
        sessions = list_participant_sessions("E4")
        session = load_nurse_stress_session(sessions[0], events=events)
        assert session.sample_rates_hz == {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}
        assert session.bvp.shape[0] > 0
        assert session.acc.shape[1] == 3

    def test_labels_are_binary_or_unlabeled(self, events):
        session = load_nurse_stress_session(list_participant_sessions("E4")[0], events=events)
        for arr in session.labels.values():
            assert set(np.unique(arr).tolist()) <= {UNLABELED, 0, 1}

    def test_some_session_overlaps_a_reported_event(self, events):
        # E4's early sessions are known to overlap reported stress events (verified during build).
        found_label = False
        for sid in list_participant_sessions("E4")[:20]:
            session = load_nurse_stress_session(sid, events=events)
            if set(np.unique(session.labels["EDA"]).tolist()) - {UNLABELED}:
                found_label = True
                break
        assert found_label

    def test_converts_to_bundle(self, events):
        session = load_nurse_stress_session(list_participant_sessions("E4")[0], events=events)
        bundle = nurse_stress_session_to_bundle(session)
        assert bundle.dataset == "nurse_stress"
        assert bundle.subject_id.startswith("E4_")
        assert bundle.channel_names == ["ACC", "BVP", "EDA", "TEMP"]

    def test_missing_session_raises(self, events):
        with pytest.raises(FileNotFoundError):
            load_nurse_stress_session("E4_9999999999", events=events)
