"""Generate a synthetic raw event stream matching docs/contracts.md (``neuroshield.hw.v1``).

Used to drive the whole software-first MVP (features, baseline, motion gate, model, dashboard)
before any hardware exists. The output is deterministic for a given seed, so a fixture generated
once and committed under ``data/fixtures/`` reproduces exactly if regenerated.

Phase model: the session is divided into named phases (quiet baseline, mild stress-like rise,
motion burst, recovery, sensor fault). Each phase drives two continuous [0, 1] curves -- a
"stress" arousal level (shared across heart rate, EDA, and temperature) and a "motion" intensity
level (drives IMU dynamics) -- plus a boolean fault mask that degrades the PPG channel during the
sensor-fault phase, exactly as illustrated in docs/contracts.md's fault example.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "neuroshield.hw.v1"
SOURCE_NAME = "synthetic"

DEFAULT_RATES_HZ = {"ppg": 64.0, "eda": 4.0, "temp": 4.0, "imu": 32.0, "health": 1.0}

# (phase_name, fraction_of_total_duration); must sum to 1.0. Callers can pass their own schedule
# (any ordered list of (name, fraction) pairs, repeats allowed) for a controllable phase schedule.
DEFAULT_PHASES = [
    ("quiet_baseline", 0.30),
    ("mild_stress_rise", 0.20),
    ("motion_burst", 0.15),
    ("recovery", 0.20),
    ("sensor_fault", 0.15),
]

# Where each named phase drives the shared "stress" curve toward. None means "hold whatever level
# the previous phase ended at" (used by motion_burst so a burst doesn't itself reset arousal).
_STRESS_TARGET = {
    "quiet_baseline": 0.0,
    "mild_stress_rise": 1.0,
    "recovery": 0.0,
    "sensor_fault": 0.0,
}

_FAULT_PHASE_NAME = "sensor_fault"
_FAULT_CHANNEL = "ppg"
_FAULT_ERROR_CODE = "sensor_disconnected"
_MOTION_PHASE_NAME = "motion_burst"
_MOTION_RAMP_S = 2.0

_BASELINE_HR_BPM = 68.0
_PEAK_HR_BPM = 100.0
_PPG_ADC_MIDPOINT = 2048
_PPG_ADC_AMPLITUDE = 300.0

_BASELINE_EDA_LEVEL = 0.25
_STRESS_EDA_GAIN = 0.35

_BASELINE_TEMP_C = 33.0

_GRAVITY_MS2 = np.array([0.0, 9.81, 0.0])


@dataclass
class PhaseWindow:
    name: str
    start_s: float
    end_s: float


def resolve_phase_schedule(duration_sec: float, phases: list[tuple[str, float]] = None) -> list[PhaseWindow]:
    phases = phases or DEFAULT_PHASES
    total_fraction = sum(frac for _, frac in phases)
    windows = []
    t = 0.0
    for name, fraction in phases:
        span = duration_sec * (fraction / total_fraction)
        windows.append(PhaseWindow(name=name, start_s=t, end_s=t + span))
        t += span
    if windows:
        windows[-1].end_s = duration_sec  # absorb rounding
    return windows


def _stress_curve(phase_bounds: list[PhaseWindow]):
    times = [0.0]
    levels = [0.0]
    level = 0.0
    for phase in phase_bounds:
        target = _STRESS_TARGET.get(phase.name, level)  # unlisted / motion_burst: hold
        times.append(phase.end_s)
        levels.append(target)
        level = target
    times_arr, levels_arr = np.asarray(times), np.asarray(levels)

    def curve(t: np.ndarray) -> np.ndarray:
        return np.interp(t, times_arr, levels_arr)

    return curve


def _motion_curve(phase_bounds: list[PhaseWindow], ramp_s: float = _MOTION_RAMP_S):
    times = [0.0]
    levels = [0.0]
    for phase in phase_bounds:
        if phase.name == _MOTION_PHASE_NAME:
            span = phase.end_s - phase.start_s
            ramp = min(ramp_s, span / 2.0) if span > 0 else 0.0
            times += [phase.start_s + ramp, phase.end_s - ramp, phase.end_s]
            levels += [1.0, 1.0, 0.0]
        else:
            times.append(phase.end_s)
            levels.append(0.0)
    times_arr, levels_arr = np.asarray(times), np.asarray(levels)

    def curve(t: np.ndarray) -> np.ndarray:
        return np.interp(t, times_arr, levels_arr)

    return curve


def _fault_mask(phase_bounds: list[PhaseWindow], t: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(t, dtype=bool)
    for phase in phase_bounds:
        if phase.name == _FAULT_PHASE_NAME:
            mask |= (t >= phase.start_s) & (t < phase.end_s)
    return mask


def _synthesize_ppg(t_s: np.ndarray, stress: np.ndarray, motion: np.ndarray, fault: np.ndarray, rng) -> np.ndarray:
    hr_bpm = _BASELINE_HR_BPM + (_PEAK_HR_BPM - _BASELINE_HR_BPM) * stress
    phase = np.cumsum(hr_bpm / 60.0) * np.mean(np.diff(t_s, prepend=t_s[0] - 1.0 / 64.0))
    waveform = np.sin(2 * np.pi * phase) + 0.3 * np.sin(4 * np.pi * phase)
    noise_std = 0.03 + 0.5 * motion  # motion artifacts dominate the signal during a burst
    waveform = waveform + rng.normal(0, noise_std, size=t_s.shape)
    raw = _PPG_ADC_MIDPOINT + _PPG_ADC_AMPLITUDE * waveform
    raw = np.clip(raw, 0, 4095)
    raw[fault] = _PPG_ADC_MIDPOINT  # dead/flat signal while the channel is "disconnected"
    return raw.astype(np.int64)


def _synthesize_eda(t_s: np.ndarray, stress: np.ndarray, rate_hz: float, rng) -> np.ndarray:
    drift = np.cumsum(rng.normal(0, 0.001, size=t_s.shape))
    level = _BASELINE_EDA_LEVEL + drift + _STRESS_EDA_GAIN * stress

    # Skin-conductance-response bumps: more frequent when stress is elevated.
    scr_rate_hz = 0.01 + 0.05 * np.mean(stress)  # expected events/sec, roughly
    n_candidates = max(1, int(scr_rate_hz * t_s[-1])) if len(t_s) else 0
    onsets = rng.choice(len(t_s), size=min(n_candidates, len(t_s)), replace=False) if n_candidates else []
    for onset_idx in onsets:
        amplitude = rng.uniform(0.05, 0.25)
        tail = np.arange(len(t_s) - onset_idx) / rate_hz
        level[onset_idx:] += amplitude * np.exp(-tail / 2.5)

    return np.clip(level, 0.0, 1.0)


def _synthesize_temp(t_s: np.ndarray, stress: np.ndarray, motion: np.ndarray, rng) -> np.ndarray:
    drift = np.cumsum(rng.normal(0, 0.0015, size=t_s.shape))
    # Peripheral vasoconstriction under stress and evaporative cooling during motion both cool the skin.
    return _BASELINE_TEMP_C + drift - 0.3 * stress - 0.5 * motion


def _synthesize_imu(t_s: np.ndarray, motion: np.ndarray, rng) -> np.ndarray:
    jitter_std = 0.05 + 0.05 * motion
    dynamic_std = 4.0 * motion
    noise = rng.normal(0, 1, size=(len(t_s), 3)) * jitter_std[:, None]
    burst = rng.normal(0, 1, size=(len(t_s), 3)) * dynamic_std[:, None]
    return _GRAVITY_MS2[None, :] + noise + burst


def _channel_events(name: str, t_s: np.ndarray, payload_rows: list[dict], ok: np.ndarray, error: list) -> list[dict]:
    events = []
    for i in range(len(t_s)):
        event = {"type": name, "t_us": int(round(t_s[i] * 1e6)), "ok": bool(ok[i])}
        if ok[i]:
            event.update(payload_rows[i])
        elif error[i] is not None:
            event["error"] = error[i]
        events.append(event)
    return events


def generate_events(
    duration_sec: float,
    seed: int = 0,
    session_id: str = "synthetic-demo",
    rates_hz: dict[str, float] = None,
    phases: list[tuple[str, float]] = None,
) -> list[dict]:
    """Return the full ordered (but not yet seq-numbered) list of raw event dicts for a session."""
    rates_hz = rates_hz or DEFAULT_RATES_HZ
    phase_bounds = resolve_phase_schedule(duration_sec, phases)
    rng = np.random.default_rng(seed)

    stress_curve = _stress_curve(phase_bounds)
    motion_curve = _motion_curve(phase_bounds)

    all_events: list[dict] = []

    # PPG
    n_ppg = int(duration_sec * rates_hz["ppg"])
    t_ppg = np.arange(n_ppg) / rates_hz["ppg"]
    stress_ppg, motion_ppg = stress_curve(t_ppg), motion_curve(t_ppg)
    fault_ppg = _fault_mask(phase_bounds, t_ppg) if _FAULT_CHANNEL == "ppg" else np.zeros_like(t_ppg, dtype=bool)
    ppg_raw = _synthesize_ppg(t_ppg, stress_ppg, motion_ppg, fault_ppg, rng)
    ppg_payload = [{"ppg_raw": int(v)} for v in ppg_raw]
    ppg_ok = ~fault_ppg
    ppg_error = [_FAULT_ERROR_CODE if f else None for f in fault_ppg]
    all_events += _channel_events("ppg", t_ppg, ppg_payload, ppg_ok, ppg_error)

    # EDA
    n_eda = int(duration_sec * rates_hz["eda"])
    t_eda = np.arange(n_eda) / rates_hz["eda"]
    eda_level = _synthesize_eda(t_eda, stress_curve(t_eda), rates_hz["eda"], rng)
    eda_payload = [{"eda_level": round(float(v), 4), "eda_unit": "relative"} for v in eda_level]
    eda_ok = np.ones(n_eda, dtype=bool)
    all_events += _channel_events("eda", t_eda, eda_payload, eda_ok, [None] * n_eda)

    # TEMP
    n_temp = int(duration_sec * rates_hz["temp"])
    t_temp = np.arange(n_temp) / rates_hz["temp"]
    temp_c = _synthesize_temp(t_temp, stress_curve(t_temp), motion_curve(t_temp), rng)
    temp_payload = [{"temp_c": round(float(v), 3)} for v in temp_c]
    temp_ok = np.ones(n_temp, dtype=bool)
    all_events += _channel_events("temp", t_temp, temp_payload, temp_ok, [None] * n_temp)

    # IMU
    n_imu = int(duration_sec * rates_hz["imu"])
    t_imu = np.arange(n_imu) / rates_hz["imu"]
    acc = _synthesize_imu(t_imu, motion_curve(t_imu), rng)
    imu_payload = [
        {"acc_x": round(float(a[0]), 4), "acc_y": round(float(a[1]), 4), "acc_z": round(float(a[2]), 4)}
        for a in acc
    ]
    imu_ok = np.ones(n_imu, dtype=bool)
    all_events += _channel_events("imu", t_imu, imu_payload, imu_ok, [None] * n_imu)

    # HEALTH (~1 Hz)
    n_health = max(1, int(duration_sec * rates_hz["health"]))
    t_health = np.arange(n_health) / rates_hz["health"]
    fault_health = _fault_mask(phase_bounds, t_health)
    battery = np.clip(95.0 - 0.5 * (t_health / max(duration_sec, 1.0)) * 100.0, 5.0, 100.0)
    for i, t in enumerate(t_health):
        channels_status = {"ppg": not bool(fault_health[i]), "eda": True, "temp": True, "imu": True}
        all_events.append(
            {
                "type": "health",
                "t_us": int(round(t * 1e6)),
                "ok": True,
                "battery_pct": round(float(battery[i]), 1),
                "channels": channels_status,
                "link_quality": None,
                "uptime_s": round(float(t), 3),
                "fault": _FAULT_ERROR_CODE if fault_health[i] else None,
            }
        )

    all_events.sort(key=lambda e: e["t_us"])

    for seq, event in enumerate(all_events):
        event["schema_version"] = SCHEMA_VERSION
        event["source"] = SOURCE_NAME
        event["session_id"] = session_id
        event["seq"] = seq
        # Reorder keys so common fields read first, matching docs/contracts.md examples.
        ordered = {
            "schema_version": event.pop("schema_version"),
            "type": event.pop("type"),
            "source": event.pop("source"),
            "session_id": event.pop("session_id"),
            "seq": event.pop("seq"),
            "t_us": event.pop("t_us"),
        }
        ok_value = event.pop("ok")
        ordered.update(event)
        ordered["ok"] = ok_value
        all_events[seq] = ordered

    return all_events


def write_ndjson(events: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic neuroshield.hw.v1 NDJSON fixture")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--session-id", type=str, default="synthetic-demo")
    parser.add_argument("--rate-ppg", type=float, default=DEFAULT_RATES_HZ["ppg"])
    parser.add_argument("--rate-eda", type=float, default=DEFAULT_RATES_HZ["eda"])
    parser.add_argument("--rate-temp", type=float, default=DEFAULT_RATES_HZ["temp"])
    parser.add_argument("--rate-imu", type=float, default=DEFAULT_RATES_HZ["imu"])
    parser.add_argument("--rate-health", type=float, default=DEFAULT_RATES_HZ["health"])
    args = parser.parse_args()

    rates_hz = {
        "ppg": args.rate_ppg,
        "eda": args.rate_eda,
        "temp": args.rate_temp,
        "imu": args.rate_imu,
        "health": args.rate_health,
    }
    events = generate_events(
        duration_sec=args.duration_sec, seed=args.seed, session_id=args.session_id, rates_hz=rates_hz
    )
    write_ndjson(events, args.out)
    print(f"wrote {len(events)} events -> {args.out}")


if __name__ == "__main__":
    main()
