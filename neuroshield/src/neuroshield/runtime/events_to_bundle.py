"""Bridge from raw contract events (docs/contracts.md) to the dataset-agnostic SignalBundle.

Unlike the WESAD loader, live/replay events don't arrive on a perfectly uniform sample grid --
sensor-fault events are dropped rather than interpolated, so each channel's ``time_s`` is built
from the events' own ``t_us`` timestamps rather than assumed from index/rate. This is what lets a
gap (e.g. a faulty PPG channel) show up honestly as a time gap instead of a fabricated flat
signal.
"""

from __future__ import annotations

import numpy as np

from neuroshield.data.bundle import SignalBundle

# Raw contract event "type" -> SignalBundle channel name, matching the WESAD wrist channel
# naming used throughout data/features (BVP is the PPG-derived channel name WESAD itself uses).
CONTRACT_TYPE_TO_CHANNEL = {"ppg": "BVP", "eda": "EDA", "temp": "TEMP", "imu": "ACC"}

# Nominal rates from docs/contracts.md's target rates (== WESAD wrist native rates), used as the
# bundle's sample_rates_hz metadata. Actual per-sample timing still comes from each event's t_us.
NOMINAL_RATES_HZ = {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}

UNLABELED = -1


def events_to_bundle(events: list[dict], dataset: str = "live", subject_id: str = "live") -> SignalBundle:
    """Convert validated raw events (e.g. from ReplaySource) into a SignalBundle.

    Only ``ok=True`` events contribute samples; a channel with a sensor fault simply has fewer
    samples covering that time range, which is real signal for the quality/abstention logic, not
    something to paper over here.
    """
    buffers: dict[str, dict[str, list]] = {ch: {"t": [], "v": []} for ch in NOMINAL_RATES_HZ}

    for event in events:
        event_type = event.get("type")
        channel = CONTRACT_TYPE_TO_CHANNEL.get(event_type)
        if channel is None or not event.get("ok", False):
            continue

        t_s = event["t_us"] / 1_000_000.0
        if channel == "BVP":
            value = event.get("ppg_raw")
        elif channel == "EDA":
            value = event.get("eda_level")
        elif channel == "TEMP":
            value = event.get("temp_c")
        else:  # ACC
            value = (event.get("acc_x"), event.get("acc_y"), event.get("acc_z"))

        if value is None or (isinstance(value, tuple) and any(v is None for v in value)):
            continue

        buffers[channel]["t"].append(t_s)
        buffers[channel]["v"].append(value)

    channels, time_s, labels, sample_rates_hz = {}, {}, {}, {}
    for channel, buf in buffers.items():
        if not buf["t"]:
            continue
        values = np.asarray(buf["v"], dtype=np.float64)
        if channel == "ACC":
            values = values.reshape(-1, 3)
        channels[channel] = values
        time_s[channel] = np.asarray(buf["t"], dtype=np.float64)
        labels[channel] = np.full(len(values), UNLABELED, dtype=np.int64)
        sample_rates_hz[channel] = NOMINAL_RATES_HZ[channel]

    if not channels:
        raise ValueError("events_to_bundle: no usable ok=true events found for any known channel")

    return SignalBundle(
        dataset=dataset,
        subject_id=subject_id,
        channels=channels,
        time_s=time_s,
        sample_rates_hz=sample_rates_hz,
        labels=labels,
    )
