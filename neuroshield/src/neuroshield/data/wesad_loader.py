"""Loader for raw WESAD subject pickle files.

WESAD ships one pickle per subject (``SX/SX.pkl``) with this documented structure
(Schmidt et al., ICMI 2018; consistent across public re-implementations):

    {
        "subject": "S2",
        "signal": {
            "wrist": {"ACC": (N,3), "BVP": (N,1), "EDA": (N,1), "TEMP": (N,1)},
            "chest": {"ACC": ..., "ECG": ..., "EMG": ..., "EDA": ..., "Temp": ..., "Resp": ...},
        },
        "label": (M,) int array at 700 Hz, aligned with the chest signal,
    }

Wrist native sample rates: ACC 32 Hz, BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz. The label array is
recorded at the chest device's 700 Hz and must be resampled (nearest-index) onto each wrist
channel's own rate, since chest and wrist recordings share a synchronized start time.

Label codes (from the WESAD readme): 0 not defined/transient, 1 baseline, 2 stress,
3 amusement, 4 meditation, 5-7 unused/ignore conditions.

This structure was assumed from the public WESAD documentation before the raw archive was
fully downloaded; run this module's ``__main__`` block against a real subject file to confirm
the keys match before relying on it (see T4's "expected obstacles" note).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WRIST_SAMPLE_RATES_HZ = {
    "ACC": 32.0,
    "BVP": 64.0,
    "EDA": 4.0,
    "TEMP": 4.0,
}

CHEST_LABEL_RATE_HZ = 700.0

LABEL_NAMES = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore_5",
    6: "ignore_6",
    7: "ignore_7",
}

DEFAULT_WESAD_ROOT = Path("data/external/wesad")


@dataclass
class WesadSubjectRaw:
    subject_id: str
    bvp: np.ndarray
    eda: np.ndarray
    temp: np.ndarray
    acc: np.ndarray
    labels: dict[str, np.ndarray]
    sample_rates_hz: dict[str, float]


def _resample_labels_to_rate(label_700hz: np.ndarray, n_samples: int, target_hz: float) -> np.ndarray:
    """Map the 700 Hz chest-aligned label array onto a wrist channel's own sample grid."""
    ratio = CHEST_LABEL_RATE_HZ / target_hz
    idx = (np.arange(n_samples) * ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(label_700hz) - 1)
    return label_700hz[idx]


def load_wesad_subject(subject_id: str, root: Path = DEFAULT_WESAD_ROOT) -> WesadSubjectRaw:
    """Load one WESAD subject's wrist signals, labels (resampled per-channel), and rates.

    ``subject_id`` is e.g. ``"S2"``. Expects ``root/SX/SX.pkl`` to exist (the layout produced
    by extracting the official WESAD.zip).
    """
    pkl_path = root / subject_id / f"{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"WESAD subject pickle not found at {pkl_path}. "
            "Download and extract WESAD.zip into data/external/wesad/ first (see docs/datasets.md)."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]
    label_700hz = np.asarray(data["label"]).reshape(-1)

    bvp = np.asarray(wrist["BVP"]).reshape(-1)
    eda = np.asarray(wrist["EDA"]).reshape(-1)
    temp = np.asarray(wrist["TEMP"]).reshape(-1)
    acc = np.asarray(wrist["ACC"]).reshape(-1, 3)

    labels = {
        "BVP": _resample_labels_to_rate(label_700hz, len(bvp), WRIST_SAMPLE_RATES_HZ["BVP"]),
        "EDA": _resample_labels_to_rate(label_700hz, len(eda), WRIST_SAMPLE_RATES_HZ["EDA"]),
        "TEMP": _resample_labels_to_rate(label_700hz, len(temp), WRIST_SAMPLE_RATES_HZ["TEMP"]),
        "ACC": _resample_labels_to_rate(label_700hz, acc.shape[0], WRIST_SAMPLE_RATES_HZ["ACC"]),
    }

    return WesadSubjectRaw(
        subject_id=str(data.get("subject", subject_id)),
        bvp=bvp,
        eda=eda,
        temp=temp,
        acc=acc,
        labels=labels,
        sample_rates_hz=dict(WRIST_SAMPLE_RATES_HZ),
    )


if __name__ == "__main__":
    import sys

    sid = sys.argv[1] if len(sys.argv) > 1 else "S2"
    subject = load_wesad_subject(sid)
    print(f"subject={subject.subject_id}")
    for name, arr in (("BVP", subject.bvp), ("EDA", subject.eda), ("TEMP", subject.temp), ("ACC", subject.acc)):
        print(f"  {name}: shape={arr.shape} dtype={arr.dtype} rate={subject.sample_rates_hz.get(name)}Hz")
    print(f"  label unique values: {sorted(set(subject.labels['EDA'].tolist()))}")
