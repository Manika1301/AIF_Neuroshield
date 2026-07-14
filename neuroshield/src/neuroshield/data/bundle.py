"""Dataset-agnostic signal container used by every downstream stage (features, model, runtime).

No code outside ``src/neuroshield/data/`` should need to know how any individual dataset (WESAD,
PPG-DaLiA, Stress-Predict, ...) stores its raw signals. Loaders convert their dataset-specific
structure into a :class:`SignalBundle` once, here, and everything else works against that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignalBundle:
    dataset: str
    subject_id: str
    channels: dict[str, np.ndarray]
    time_s: dict[str, np.ndarray]
    sample_rates_hz: dict[str, float]
    labels: dict[str, np.ndarray]
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("SignalBundle.dataset must be a non-empty string")
        if not self.subject_id:
            raise ValueError("SignalBundle.subject_id must be a non-empty string")
        if not self.channels:
            raise ValueError("SignalBundle.channels must not be empty")

        channel_names = set(self.channels)
        for field_name, mapping in (
            ("time_s", self.time_s),
            ("sample_rates_hz", self.sample_rates_hz),
            ("labels", self.labels),
        ):
            missing = channel_names - set(mapping)
            if missing:
                raise ValueError(f"SignalBundle.{field_name} is missing channels: {sorted(missing)}")

        for name, arr in self.channels.items():
            arr = np.asarray(arr)
            if arr.shape[0] == 0:
                raise ValueError(f"SignalBundle channel {name!r} is empty")

            n = arr.shape[0]
            t = np.asarray(self.time_s[name])
            if t.shape[0] != n:
                raise ValueError(
                    f"SignalBundle.time_s[{name!r}] has length {t.shape[0]}, expected {n}"
                )
            if t.shape[0] > 1 and np.any(np.diff(t) < 0):
                raise ValueError(f"SignalBundle.time_s[{name!r}] is not monotonic non-decreasing")

            lab = np.asarray(self.labels[name])
            if lab.shape[0] != n:
                raise ValueError(
                    f"SignalBundle.labels[{name!r}] has length {lab.shape[0]}, expected {n}"
                )

            if self.sample_rates_hz[name] <= 0:
                raise ValueError(f"SignalBundle.sample_rates_hz[{name!r}] must be positive")

    @property
    def channel_names(self) -> list[str]:
        return sorted(self.channels)


def from_channel_arrays(
    dataset: str,
    subject_id: str,
    channels: dict[str, np.ndarray],
    sample_rates_hz: dict[str, float],
    labels: dict[str, np.ndarray],
    warnings: list[str] | None = None,
) -> SignalBundle:
    """Build a SignalBundle, deriving per-channel time arrays from each channel's sample rate."""
    time_s = {
        name: np.arange(np.asarray(arr).shape[0], dtype=np.float64) / sample_rates_hz[name]
        for name, arr in channels.items()
    }
    return SignalBundle(
        dataset=dataset,
        subject_id=subject_id,
        channels=channels,
        time_s=time_s,
        sample_rates_hz=dict(sample_rates_hz),
        labels=labels,
        warnings=list(warnings or []),
    )


def wesad_subject_to_bundle(subject) -> SignalBundle:  # noqa: ANN001 - see wesad_loader.WesadSubjectRaw
    """Convert a raw WESAD wrist subject (from wesad_loader.load_wesad_subject) into a SignalBundle."""
    channels = {
        "BVP": subject.bvp,
        "EDA": subject.eda,
        "TEMP": subject.temp,
        "ACC": subject.acc,
    }
    return from_channel_arrays(
        dataset="wesad",
        subject_id=subject.subject_id,
        channels=channels,
        sample_rates_hz=subject.sample_rates_hz,
        labels=subject.labels,
    )


def stress_predict_subject_to_bundle(subject) -> SignalBundle:  # noqa: ANN001 - see stress_predict_loader.StressPredictSubjectRaw
    """Convert a raw Stress-Predict wrist subject into a SignalBundle."""
    channels = {
        "BVP": subject.bvp,
        "EDA": subject.eda,
        "TEMP": subject.temp,
        "ACC": subject.acc,
    }
    return from_channel_arrays(
        dataset="stress_predict",
        subject_id=subject.subject_id,
        channels=channels,
        sample_rates_hz=subject.sample_rates_hz,
        labels=subject.labels,
    )


def nurse_stress_session_to_bundle(session) -> SignalBundle:  # noqa: ANN001 - see nurse_stress_loader.NurseStressSessionRaw
    """Convert a raw Nurse Stress session into a SignalBundle (one bundle per session).

    subject_id is set to the session id (``<ID>_<unix_start>``) so downstream grouping can recover
    the participant via ``session_id.split("_")[0]`` while keeping sessions individually addressable.
    """
    channels = {
        "BVP": session.bvp,
        "EDA": session.eda,
        "TEMP": session.temp,
        "ACC": session.acc,
    }
    return from_channel_arrays(
        dataset="nurse_stress",
        subject_id=session.session_id,
        channels=channels,
        sample_rates_hz=session.sample_rates_hz,
        labels=session.labels,
    )
