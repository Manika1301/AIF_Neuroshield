"""Convert a SignalBundle into fixed-schema, windowed feature rows (``features-v2``).

Windows are 60 seconds with 30-second overlap (step = 30s) by default, matching T6. Every
feature column is always present in the output, even when a value could not be computed for a
given window (it is then NaN, not silently dropped) -- so downstream code can rely on a stable
column set (``FEATURE_COLUMNS``) forever, and any accidental column drift is caught by
``tests/test_features.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import neurokit2 as nk
import numpy as np
import pandas as pd

from neuroshield.data.bundle import SignalBundle

FEATURE_VERSION = "features-v2"

FEATURE_COLUMNS = [
    "hr_mean_bpm",
    "ibi_sd_ms",
    "ibi_rmssd_ms",
    "ppg_quality",
    # frequency-domain HRV (features-v2): sympathovagal balance, a strong stress signal
    "hrv_lf",
    "hrv_hf",
    "hrv_lf_hf_ratio",
    "eda_level",
    "eda_slope",
    "eda_response_count",
    "eda_response_mean_amp",
    # cvxEDA tonic/phasic decomposition (features-v2): sustained vs. momentary arousal
    "eda_tonic_mean",
    "eda_tonic_slope",
    "eda_phasic_mean",
    "temp_mean_c",
    "temp_slope_c_per_min",
    "motion_dynamic_rms",
    "motion_dynamic_p95",
    "valid_fraction",
]

METADATA_COLUMNS = [
    "dataset",
    "subject_id",
    "window_start_s",
    "window_end_s",
    "label",
    "feature_version",
]

ALL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

DEFAULT_WINDOW_SEC = 60.0
DEFAULT_STEP_SEC = 30.0


@dataclass
class _WindowSlice:
    values: np.ndarray
    rate_hz: float
    labels: np.ndarray


def _slice_channel(bundle: SignalBundle, name: str, start_s: float, end_s: float) -> _WindowSlice:
    t = bundle.time_s[name]
    mask = (t >= start_s) & (t < end_s)
    values = np.asarray(bundle.channels[name])[mask]
    labels = np.asarray(bundle.labels[name])[mask]
    return _WindowSlice(values=values, rate_hz=bundle.sample_rates_hz[name], labels=labels)


def _linear_slope_per_sec(values: np.ndarray, rate_hz: float) -> float:
    if len(values) < 2:
        return float("nan")
    t = np.arange(len(values)) / rate_hz
    slope, _ = np.polyfit(t, values, 1)
    return float(slope)


def _ppg_features(ppg: _WindowSlice) -> dict:
    out = {
        "hr_mean_bpm": np.nan,
        "ibi_sd_ms": np.nan,
        "ibi_rmssd_ms": np.nan,
        "ppg_quality": np.nan,
        "hrv_lf": np.nan,
        "hrv_hf": np.nan,
        "hrv_lf_hf_ratio": np.nan,
    }
    expected = int(round(DEFAULT_WINDOW_SEC * ppg.rate_hz))
    if len(ppg.values) < min(expected, ppg.rate_hz * 2):  # need at least ~2s of signal
        return out
    rate = int(round(ppg.rate_hz))
    try:
        cleaned = nk.ppg_clean(ppg.values, sampling_rate=rate)
        _, info = nk.ppg_peaks(cleaned, sampling_rate=rate)
        peak_idx = np.asarray(info["PPG_Peaks"])
        if len(peak_idx) >= 3:
            peak_times_s = peak_idx / rate
            ibi_ms = np.diff(peak_times_s) * 1000.0
            out["hr_mean_bpm"] = float(60000.0 / np.mean(ibi_ms))
            out["ibi_sd_ms"] = float(np.std(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else float("nan")
            if len(ibi_ms) > 2:
                out["ibi_rmssd_ms"] = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
            out.update(_hrv_frequency(peak_idx, rate))
        quality = nk.ppg_quality(cleaned, sampling_rate=rate)
        out["ppg_quality"] = float(np.nanmean(quality))
    except Exception:
        pass
    return out


def _hrv_frequency(peak_idx: np.ndarray, rate: int) -> dict:
    """Frequency-domain HRV (LF/HF power and ratio) -- a strong, standard stress signal.

    LF (~0.04-0.15 Hz) reflects mixed sympathetic/parasympathetic tone, HF (~0.15-0.4 Hz) mostly
    parasympathetic; the LF/HF ratio rises under stress. Needs enough beats, so it degrades to NaN
    on short/sparse windows rather than fabricating a value.
    """
    out = {"hrv_lf": np.nan, "hrv_hf": np.nan, "hrv_lf_hf_ratio": np.nan}
    if len(peak_idx) < 8:  # frequency-domain HRV is unreliable with too few beats
        return out
    try:
        freq = nk.hrv_frequency(peak_idx, sampling_rate=rate, silent=True)
        out["hrv_lf"] = float(freq["HRV_LF"].iloc[0])
        out["hrv_hf"] = float(freq["HRV_HF"].iloc[0])
        out["hrv_lf_hf_ratio"] = float(freq["HRV_LFHF"].iloc[0])
    except Exception:
        pass
    return out


def _eda_features(eda: _WindowSlice) -> dict:
    out = {
        "eda_level": np.nan,
        "eda_slope": np.nan,
        "eda_response_count": np.nan,
        "eda_response_mean_amp": np.nan,
        "eda_tonic_mean": np.nan,
        "eda_tonic_slope": np.nan,
        "eda_phasic_mean": np.nan,
    }
    if len(eda.values) < max(4, int(eda.rate_hz * 2)):
        return out
    out["eda_level"] = float(np.nanmean(eda.values))
    out["eda_slope"] = _linear_slope_per_sec(eda.values, eda.rate_hz)
    rate = int(round(eda.rate_hz))
    try:
        cleaned = nk.eda_clean(eda.values, sampling_rate=rate)
        _, info = nk.eda_peaks(cleaned, sampling_rate=rate)
        amplitudes = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
        amplitudes = amplitudes[~np.isnan(amplitudes)]
        out["eda_response_count"] = float(len(amplitudes))
        out["eda_response_mean_amp"] = float(np.mean(amplitudes)) if len(amplitudes) else 0.0
        out.update(_eda_decomposition(cleaned, rate))
    except Exception:
        pass
    return out


def _eda_decomposition(cleaned_eda: np.ndarray, rate: int) -> dict:
    """cvxEDA tonic/phasic split: tonic = sustained arousal level, phasic = momentary responses.

    Falls back to the high-pass decomposition if cvxEDA is unavailable, so features-v2 still
    produces these columns rather than all-NaN. Returns NaN only if both methods fail.
    """
    out = {"eda_tonic_mean": np.nan, "eda_tonic_slope": np.nan, "eda_phasic_mean": np.nan}
    for method in ("cvxeda", "highpass"):
        try:
            comps = nk.eda_phasic(cleaned_eda, sampling_rate=rate, method=method)
            tonic = comps["EDA_Tonic"].to_numpy()
            phasic = comps["EDA_Phasic"].to_numpy()
            out["eda_tonic_mean"] = float(np.nanmean(tonic))
            out["eda_tonic_slope"] = _linear_slope_per_sec(tonic, rate)
            out["eda_phasic_mean"] = float(np.nanmean(np.abs(phasic)))
            return out
        except Exception:
            continue
    return out


def _temp_features(temp: _WindowSlice) -> dict:
    out = {"temp_mean_c": np.nan, "temp_slope_c_per_min": np.nan}
    if len(temp.values) < 2:
        return out
    out["temp_mean_c"] = float(np.nanmean(temp.values))
    slope_per_sec = _linear_slope_per_sec(temp.values, temp.rate_hz)
    out["temp_slope_c_per_min"] = slope_per_sec * 60.0 if not np.isnan(slope_per_sec) else np.nan
    return out


def _motion_features(acc: _WindowSlice) -> dict:
    out = {"motion_dynamic_rms": np.nan, "motion_dynamic_p95": np.nan}
    if len(acc.values) < 2:
        return out
    magnitude = np.linalg.norm(acc.values, axis=1)
    dynamic = magnitude - np.mean(magnitude)  # remove the static/gravity DC component
    out["motion_dynamic_rms"] = float(np.sqrt(np.mean(dynamic**2)))
    out["motion_dynamic_p95"] = float(np.percentile(np.abs(dynamic), 95))
    return out


def _valid_fraction(bundle: SignalBundle, start_s: float, end_s: float, window_sec: float) -> float:
    fractions = []
    for name in bundle.channel_names:
        expected = max(1, round(window_sec * bundle.sample_rates_hz[name]))
        t = bundle.time_s[name]
        actual = int(np.sum((t >= start_s) & (t < end_s)))
        fractions.append(min(1.0, actual / expected))
    return float(min(fractions)) if fractions else 0.0


def _window_label(bundle: SignalBundle, start_s: float, end_s: float) -> float:
    ref_channel = "EDA" if "EDA" in bundle.channel_names else bundle.channel_names[0]
    sl = _slice_channel(bundle, ref_channel, start_s, end_s)
    if len(sl.labels) == 0:
        return np.nan
    values, counts = np.unique(sl.labels, return_counts=True)
    return float(values[np.argmax(counts)])


def extract_features(
    bundle: SignalBundle,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
) -> pd.DataFrame:
    """Slide fixed windows over every channel in ``bundle`` and compute ``FEATURE_COLUMNS``.

    A window is skipped entirely (not just filled with NaN) only if every channel has zero
    samples inside it -- there is nothing to compute and including it would be a fabricated row.
    Windowing spans the *longest* channel's coverage (not the shortest): a channel that runs out
    of data early (e.g. a sensor-fault gap) should degrade to NaN features for the windows it
    can't cover, not silently truncate every other channel's windows too.
    """
    duration_s = max(bundle.time_s[name][-1] for name in bundle.channel_names)
    rows = []
    start = 0.0
    while start + window_sec <= duration_s + 1e-9:
        end = start + window_sec

        acc_slice = _slice_channel(bundle, "ACC", start, end) if "ACC" in bundle.channels else None
        any_samples = any(
            len(_slice_channel(bundle, name, start, end).values) > 0 for name in bundle.channel_names
        )
        if not any_samples:
            start += step_sec
            continue

        row = {
            "dataset": bundle.dataset,
            "subject_id": bundle.subject_id,
            "window_start_s": start,
            "window_end_s": end,
            "label": _window_label(bundle, start, end),
            "feature_version": FEATURE_VERSION,
        }

        if "BVP" in bundle.channels:
            row.update(_ppg_features(_slice_channel(bundle, "BVP", start, end)))
        else:
            row.update({k: np.nan for k in (
                "hr_mean_bpm", "ibi_sd_ms", "ibi_rmssd_ms", "ppg_quality",
                "hrv_lf", "hrv_hf", "hrv_lf_hf_ratio",
            )})

        if "EDA" in bundle.channels:
            row.update(_eda_features(_slice_channel(bundle, "EDA", start, end)))
        else:
            row.update({k: np.nan for k in (
                "eda_level", "eda_slope", "eda_response_count", "eda_response_mean_amp",
                "eda_tonic_mean", "eda_tonic_slope", "eda_phasic_mean",
            )})

        if "TEMP" in bundle.channels:
            row.update(_temp_features(_slice_channel(bundle, "TEMP", start, end)))
        else:
            row.update({k: np.nan for k in ("temp_mean_c", "temp_slope_c_per_min")})

        if acc_slice is not None:
            row.update(_motion_features(acc_slice))
        else:
            row.update({k: np.nan for k in ("motion_dynamic_rms", "motion_dynamic_p95")})

        row["valid_fraction"] = _valid_fraction(bundle, start, end, window_sec)

        rows.append(row)
        start += step_sec

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from neuroshield.data.bundle import wesad_subject_to_bundle
    from neuroshield.data.wesad_loader import load_wesad_subject

    subject_id = sys.argv[1] if len(sys.argv) > 1 else "S2"
    subject = load_wesad_subject(subject_id)
    bundle = wesad_subject_to_bundle(subject)
    df = extract_features(bundle)

    out_dir = Path("data/interim/wesad_features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} windows x {len(df.columns)} columns -> {out_path}")
    print(df.head())
