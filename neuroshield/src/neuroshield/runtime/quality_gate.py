"""Motion and signal-quality abstention: decide when M1 must not produce a confident status.

Abstention is checked before, and kept entirely separate from, the green/amber/red decision
(T14). A window that abstains never gets a color status -- it gets ``motion_paused`` or
``poor_signal`` instead, regardless of what the model would have predicted.

Thresholds below are provisional, calibrated against this project's own synthetic motion-burst
phase (quiet ~0.05 rms / 0.10 p95 vs. motion burst ~3.6 rms / 6.9 p95 -- see T13 dev notes) with a
wide safety margin. O1 (PPG-DaLiA motion-quality analysis) replaces these with evidence from real
wrist-PPG-under-motion data; nothing downstream should need to change when that happens, only
these constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from neuroshield.features.labels import DEFAULT_MIN_VALID_FRACTION

MOTION_DYNAMIC_RMS_MAX = 1.0  # m/s^2
MOTION_DYNAMIC_P95_MAX = 2.0  # m/s^2
PPG_QUALITY_MIN = 0.7
VALID_FRACTION_MIN = DEFAULT_MIN_VALID_FRACTION  # 0.9, consistent with M1 training's own gate

MOTION_PAUSED = "motion_paused"
POOR_SIGNAL = "poor_signal"


@dataclass
class AbstentionResult:
    abstain: bool
    reason: str | None  # MOTION_PAUSED, POOR_SIGNAL, or None
    triggers: list[str] = field(default_factory=list)


def _is_bad(value) -> bool:
    return value is None or pd.isna(value)


def check_abstention(row) -> AbstentionResult:
    """Evaluate one feature window (dict-like with FEATURE_COLUMNS keys) for abstention.

    Motion is checked first: a motion-driven quality drop is reported as ``motion_paused``, not
    ``poor_signal``, so the reason points at the actual cause. Anything else that leaves signal
    coverage or PPG quality unreliable -- including simply not being able to measure motion or
    quality at all -- falls back to ``poor_signal``.
    """
    motion_rms = row["motion_dynamic_rms"]
    motion_p95 = row["motion_dynamic_p95"]
    ppg_quality = row["ppg_quality"]
    valid_fraction = row["valid_fraction"]

    motion_triggers = []
    if not _is_bad(motion_rms) and motion_rms > MOTION_DYNAMIC_RMS_MAX:
        motion_triggers.append(f"motion_dynamic_rms={motion_rms:.3f}>{MOTION_DYNAMIC_RMS_MAX}")
    if not _is_bad(motion_p95) and motion_p95 > MOTION_DYNAMIC_P95_MAX:
        motion_triggers.append(f"motion_dynamic_p95={motion_p95:.3f}>{MOTION_DYNAMIC_P95_MAX}")
    if motion_triggers:
        return AbstentionResult(abstain=True, reason=MOTION_PAUSED, triggers=motion_triggers)

    quality_triggers = []
    if _is_bad(valid_fraction) or valid_fraction < VALID_FRACTION_MIN:
        quality_triggers.append(f"valid_fraction={valid_fraction!r}<{VALID_FRACTION_MIN}")
    if _is_bad(ppg_quality) or ppg_quality < PPG_QUALITY_MIN:
        quality_triggers.append(f"ppg_quality={ppg_quality!r}<{PPG_QUALITY_MIN}")
    if _is_bad(motion_rms) or _is_bad(motion_p95):
        quality_triggers.append("motion could not be assessed")
    if quality_triggers:
        return AbstentionResult(abstain=True, reason=POOR_SIGNAL, triggers=quality_triggers)

    return AbstentionResult(abstain=False, reason=None, triggers=[])


def annotate_abstention(features: pd.DataFrame) -> pd.DataFrame:
    """Add ``abstain`` and ``abstention_reason`` columns, one row's worth of check_abstention each."""
    results = [check_abstention(row) for _, row in features.iterrows()]
    out = features.copy()
    out["abstain"] = [r.abstain for r in results]
    out["abstention_reason"] = [r.reason for r in results]
    return out
