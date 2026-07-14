import numpy as np
import pytest

from neuroshield.runtime.events_to_bundle import events_to_bundle
from neuroshield.runtime.synthetic_source import generate_events


class TestEventsToBundle:
    def test_builds_all_four_channels_when_no_faults(self):
        events = generate_events(duration_sec=10.0, seed=1, phases=[("quiet_baseline", 1.0)])
        bundle = events_to_bundle(events, dataset="synthetic", subject_id="s1")
        assert set(bundle.channel_names) == {"ACC", "BVP", "EDA", "TEMP"}

    def test_channel_lengths_match_ok_event_counts(self):
        events = generate_events(duration_sec=10.0, seed=1, phases=[("quiet_baseline", 1.0)])
        bundle = events_to_bundle(events)
        n_ppg_ok = sum(1 for e in events if e["type"] == "ppg" and e["ok"])
        assert len(bundle.channels["BVP"]) == n_ppg_ok

    def test_time_s_is_monotonic_per_channel(self):
        events = generate_events(duration_sec=10.0, seed=1, phases=[("quiet_baseline", 1.0)])
        bundle = events_to_bundle(events)
        for name in bundle.channel_names:
            t = bundle.time_s[name]
            assert np.all(np.diff(t) >= 0)

    def test_sample_rates_match_contract_targets(self):
        events = generate_events(duration_sec=10.0, seed=1, phases=[("quiet_baseline", 1.0)])
        bundle = events_to_bundle(events)
        assert bundle.sample_rates_hz == {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}

    def test_labels_are_unlabeled_placeholder(self):
        events = generate_events(duration_sec=5.0, seed=1, phases=[("quiet_baseline", 1.0)])
        bundle = events_to_bundle(events)
        for name in bundle.channel_names:
            assert set(bundle.labels[name].tolist()) == {-1}

    def test_sensor_fault_shrinks_only_the_faulty_channel(self):
        # Half the session is quiet (ppg healthy), half has a sensor fault (ppg unhealthy), so
        # the BVP channel ends up shorter than the raw ppg event count, but EDA stays complete.
        events = generate_events(
            duration_sec=10.0, seed=2, phases=[("quiet_baseline", 0.5), ("sensor_fault", 0.5)]
        )
        bundle = events_to_bundle(events)
        n_ppg_total = sum(1 for e in events if e["type"] == "ppg")
        n_eda_total = sum(1 for e in events if e["type"] == "eda")
        assert len(bundle.channels["BVP"]) < n_ppg_total  # ppg faulted -> fewer samples than raw events
        assert len(bundle.channels["EDA"]) == n_eda_total  # eda unaffected

    def test_acc_channel_has_three_columns(self):
        events = generate_events(duration_sec=5.0, seed=1, phases=[("quiet_baseline", 1.0)])
        bundle = events_to_bundle(events)
        assert bundle.channels["ACC"].shape[1] == 3

    def test_raises_on_no_usable_events(self):
        health_only = [e for e in generate_events(duration_sec=5.0, seed=1) if e["type"] == "health"]
        with pytest.raises(ValueError, match="no usable"):
            events_to_bundle(health_only)
