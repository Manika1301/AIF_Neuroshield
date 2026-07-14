import numpy as np
import pytest

from neuroshield.data.bundle import SignalBundle, from_channel_arrays, wesad_subject_to_bundle
from neuroshield.data.wesad_loader import DEFAULT_WESAD_ROOT, load_wesad_subject

S2_PICKLE = DEFAULT_WESAD_ROOT / "S2" / "S2.pkl"
requires_wesad_s2 = pytest.mark.skipif(
    not S2_PICKLE.exists(),
    reason=f"WESAD subject file not found at {S2_PICKLE}; download WESAD to run this test",
)


def _synthetic_channels():
    rng = np.random.default_rng(0)
    return {
        "BVP": rng.normal(size=640),
        "EDA": rng.normal(size=40),
        "TEMP": rng.normal(size=40) + 33.0,
        "ACC": rng.normal(size=(320, 3)),
    }


def _synthetic_rates():
    return {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}


def _synthetic_labels(channels):
    return {name: np.ones(arr.shape[0], dtype=np.int64) for name, arr in channels.items()}


class TestSignalBundleInvariants:
    """Generic invariants that hold for any dataset, checked without needing real WESAD data."""

    def test_builds_from_channel_arrays(self):
        channels = _synthetic_channels()
        bundle = from_channel_arrays(
            dataset="synthetic",
            subject_id="S99",
            channels=channels,
            sample_rates_hz=_synthetic_rates(),
            labels=_synthetic_labels(channels),
        )
        assert isinstance(bundle, SignalBundle)
        assert bundle.subject_id == "S99"
        assert bundle.channel_names == ["ACC", "BVP", "EDA", "TEMP"]

    def test_time_arrays_are_monotonic_and_non_empty(self):
        channels = _synthetic_channels()
        bundle = from_channel_arrays(
            dataset="synthetic",
            subject_id="S99",
            channels=channels,
            sample_rates_hz=_synthetic_rates(),
            labels=_synthetic_labels(channels),
        )
        for name in bundle.channel_names:
            t = bundle.time_s[name]
            assert t.shape[0] > 0
            assert np.all(np.diff(t) >= 0)

    def test_labels_present_for_every_channel(self):
        channels = _synthetic_channels()
        bundle = from_channel_arrays(
            dataset="synthetic",
            subject_id="S99",
            channels=channels,
            sample_rates_hz=_synthetic_rates(),
            labels=_synthetic_labels(channels),
        )
        for name in bundle.channel_names:
            assert name in bundle.labels
            assert len(bundle.labels[name]) == len(bundle.channels[name])

    def test_rejects_empty_channel(self):
        channels = _synthetic_channels()
        channels["EDA"] = np.array([])
        with pytest.raises(ValueError, match="empty"):
            from_channel_arrays(
                dataset="synthetic",
                subject_id="S99",
                channels=channels,
                sample_rates_hz=_synthetic_rates(),
                labels=_synthetic_labels(channels),
            )

    def test_rejects_non_monotonic_time(self):
        channels = _synthetic_channels()
        bundle = from_channel_arrays(
            dataset="synthetic",
            subject_id="S99",
            channels=channels,
            sample_rates_hz=_synthetic_rates(),
            labels=_synthetic_labels(channels),
        )
        bad_time = dict(bundle.time_s)
        bad_time["BVP"] = bad_time["BVP"][::-1]
        with pytest.raises(ValueError, match="monotonic"):
            SignalBundle(
                dataset=bundle.dataset,
                subject_id=bundle.subject_id,
                channels=bundle.channels,
                time_s=bad_time,
                sample_rates_hz=bundle.sample_rates_hz,
                labels=bundle.labels,
            )

    def test_rejects_missing_subject_id(self):
        channels = _synthetic_channels()
        with pytest.raises(ValueError, match="subject_id"):
            from_channel_arrays(
                dataset="synthetic",
                subject_id="",
                channels=channels,
                sample_rates_hz=_synthetic_rates(),
                labels=_synthetic_labels(channels),
            )


@requires_wesad_s2
class TestWesadSubjectS2:
    """Loads a real subject once WESAD.zip has been downloaded and extracted."""

    def test_subject_id_matches(self):
        subject = load_wesad_subject("S2")
        assert subject.subject_id == "S2"

    def test_expected_channel_names(self):
        subject = load_wesad_subject("S2")
        assert set(subject.sample_rates_hz) == {"BVP", "EDA", "TEMP", "ACC"}

    def test_channel_arrays_non_empty(self):
        subject = load_wesad_subject("S2")
        assert subject.bvp.shape[0] > 0
        assert subject.eda.shape[0] > 0
        assert subject.temp.shape[0] > 0
        assert subject.acc.shape[0] > 0

    def test_native_sample_rates_preserved(self):
        subject = load_wesad_subject("S2")
        assert subject.sample_rates_hz == {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}

    def test_labels_present_and_nonzero_baseline(self):
        subject = load_wesad_subject("S2")
        assert set(subject.labels) == {"BVP", "EDA", "TEMP", "ACC"}
        assert (subject.labels["EDA"] == 1).sum() > 0  # baseline windows exist

    def test_converts_to_signal_bundle(self):
        subject = load_wesad_subject("S2")
        bundle = wesad_subject_to_bundle(subject)
        assert bundle.dataset == "wesad"
        assert bundle.subject_id == "S2"
        assert bundle.channel_names == ["ACC", "BVP", "EDA", "TEMP"]
        for name in bundle.channel_names:
            t = bundle.time_s[name]
            assert np.all(np.diff(t) >= 0)
