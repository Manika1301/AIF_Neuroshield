"""Loader for the Stress-Predict Dataset (external validation, T18).

Stress-Predict ships raw Empatica E4 exports per subject (``Raw_data/SX/{ACC,BVP,EDA,TEMP}.csv``)
in the standard E4 format: line 1 is the UTC unix start time, line 2 is the sample rate in Hz,
then one row per sample (ACC.csv has three comma-separated columns for x/y/z, in units of
1/64 g -- the same raw unit WESAD's wrist ACC uses, since both datasets came off the same
Empatica hardware family).

Ground-truth labels are not in the raw per-subject folders; they come from
``Processed_data/Improved_All_Combined_hr_rsp_binary.csv``, a per-second binary label
(0 = non-stress/baseline, 1 = stress task) keyed by participant and unix timestamp. Each raw
sample is labelled by looking up the nearest whole second in that table; samples outside the
table's covered time range are labelled ``-1`` (unknown/excluded), the same convention T7 uses
for WESAD windows with no usable label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("data/external/stress_predict")
LABEL_CSV_RELATIVE_PATH = "Processed_data/Improved_All_Combined_hr_rsp_binary.csv"

UNLABELED = -1
BASELINE_LABEL = 0
STRESS_LABEL = 1


@dataclass
class StressPredictSubjectRaw:
    subject_id: str
    bvp: np.ndarray
    eda: np.ndarray
    temp: np.ndarray
    acc: np.ndarray
    labels: dict[str, np.ndarray]
    sample_rates_hz: dict[str, float]


def _read_e4_single_column_csv(path: Path) -> tuple[float, float, np.ndarray]:
    with open(path) as f:
        start_time = float(f.readline().strip())
        rate = float(f.readline().strip())
        values = np.array([float(line) for line in f if line.strip()], dtype=np.float64)
    return start_time, rate, values


def _read_e4_acc_csv(path: Path) -> tuple[float, float, np.ndarray]:
    with open(path) as f:
        start_time = float(f.readline().split(",")[0].strip())
        rate = float(f.readline().split(",")[0].strip())
        rows = [
            [float(v) for v in line.split(",")]
            for line in f
            if line.strip()
        ]
    return start_time, rate, np.array(rows, dtype=np.float64)


def _load_label_table(root: Path, subject_number: int) -> pd.Series:
    """Return a Series indexed by unix second, value in {0, 1}, for one participant."""
    label_path = root / LABEL_CSV_RELATIVE_PATH
    df = pd.read_csv(label_path)
    subject_df = df[df["Participant"] == subject_number]
    if subject_df.empty:
        return pd.Series(dtype=np.int64)
    return subject_df.set_index("Time(sec)")["Label"].astype(np.int64)


def _label_samples(timestamps_s: np.ndarray, label_table: pd.Series) -> np.ndarray:
    if len(label_table) == 0:
        return np.full(len(timestamps_s), UNLABELED, dtype=np.int64)

    known_times = label_table.index.to_numpy(dtype=np.float64)
    known_labels = label_table.to_numpy()
    order = np.argsort(known_times)
    known_times, known_labels = known_times[order], known_labels[order]

    nearest_second = np.round(timestamps_s)
    idx = np.searchsorted(known_times, nearest_second)
    idx = np.clip(idx, 0, len(known_times) - 1)
    # Check the neighbor on the left too, since searchsorted gives the insertion point.
    left_idx = np.clip(idx - 1, 0, len(known_times) - 1)
    use_left = np.abs(known_times[left_idx] - nearest_second) < np.abs(known_times[idx] - nearest_second)
    chosen_idx = np.where(use_left, left_idx, idx)

    within_one_second = np.abs(known_times[chosen_idx] - nearest_second) <= 1.0
    labels = np.where(within_one_second, known_labels[chosen_idx], UNLABELED)
    return labels.astype(np.int64)


def load_stress_predict_subject(subject_id: str, root: Path = DEFAULT_ROOT) -> StressPredictSubjectRaw:
    """Load one Stress-Predict subject's wrist signals and per-sample binary labels.

    ``subject_id`` is e.g. ``"S01"``. Expects ``root/Raw_data/SX/{ACC,BVP,EDA,TEMP}.csv`` and
    ``root/Processed_data/Improved_All_Combined_hr_rsp_binary.csv`` to exist.
    """
    subject_dir = root / "Raw_data" / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(
            f"Stress-Predict subject folder not found at {subject_dir}. "
            "Clone the dataset into data/external/stress_predict/ first (see docs/datasets.md)."
        )

    bvp_start, bvp_rate, bvp = _read_e4_single_column_csv(subject_dir / "BVP.csv")
    eda_start, eda_rate, eda = _read_e4_single_column_csv(subject_dir / "EDA.csv")
    temp_start, temp_rate, temp = _read_e4_single_column_csv(subject_dir / "TEMP.csv")
    acc_start, acc_rate, acc = _read_e4_acc_csv(subject_dir / "ACC.csv")

    subject_number = int(subject_id.lstrip("Ss"))
    label_table = _load_label_table(root, subject_number)

    labels = {
        "BVP": _label_samples(bvp_start + np.arange(len(bvp)) / bvp_rate, label_table),
        "EDA": _label_samples(eda_start + np.arange(len(eda)) / eda_rate, label_table),
        "TEMP": _label_samples(temp_start + np.arange(len(temp)) / temp_rate, label_table),
        "ACC": _label_samples(acc_start + np.arange(len(acc)) / acc_rate, label_table),
    }

    return StressPredictSubjectRaw(
        subject_id=subject_id,
        bvp=bvp,
        eda=eda,
        temp=temp,
        acc=acc,
        labels=labels,
        sample_rates_hz={"BVP": bvp_rate, "EDA": eda_rate, "TEMP": temp_rate, "ACC": acc_rate},
    )
