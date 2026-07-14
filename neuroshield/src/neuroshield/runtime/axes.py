"""D4: four physiological arousal axes derived from personal baseline z-scores.

Tier 1 of the redesign (see docs/design_doc.tex): instead of one stress number, show which of the
four measured systems is driving arousal. Each axis aggregates its features' z-scores *in the
arousal direction* -- crucially, the direction is not always "higher = more aroused":

  - cardiac:      higher heart rate = more arousal (+); higher pulse variability = less (-)
  - electrodermal: higher skin-conductance level / more responses = more arousal (+)
  - thermal:      cooler skin (peripheral vasoconstriction under stress) = more arousal (-)
  - movement:     more motion = more activity (+)  [activity, reported alongside the others]

No model: these are honest, directly-measured, personalized transforms. NaN-safe: a feature that
could not be computed for a window simply does not contribute to its axis; an axis with no usable
features reports ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Per-feature sign: +1 if a higher value indicates more arousal/activity on that axis, -1 if lower.
AXIS_FEATURE_SIGNS = {
    # higher LF/HF ratio = more sympathetic dominance = more arousal (features-v2)
    "cardiac": {"hr_mean_bpm": +1, "ibi_sd_ms": -1, "ibi_rmssd_ms": -1, "hrv_lf_hf_ratio": +1},
    "electrodermal": {
        "eda_level": +1,
        "eda_slope": +1,
        "eda_response_count": +1,
        "eda_response_mean_amp": +1,
        "eda_tonic_mean": +1,  # sustained arousal (features-v2)
        "eda_phasic_mean": +1,  # momentary responses (features-v2)
    },
    "thermal": {"temp_mean_c": -1, "temp_slope_c_per_min": -1},
    "movement": {"motion_dynamic_rms": +1, "motion_dynamic_p95": +1},
}

AXIS_NAMES = list(AXIS_FEATURE_SIGNS.keys())

# Level thresholds on the arousal-direction aggregate z-score.
ELEVATED_Z = 1.0
HIGH_Z = 2.0

# Clip aggregate scores to a sane range so one wild feature can't dominate a radar view.
SCORE_CLIP = 5.0

LEVEL_NORMAL = "normal"
LEVEL_ELEVATED = "elevated"
LEVEL_HIGH = "high"


@dataclass
class AxisScore:
    name: str
    score: float | None  # signed arousal-direction aggregate z-score (None if no usable features)
    level: str  # normal / elevated / high (None-safe: normal when score is None)
    n_features: int  # how many features contributed

    def to_dict(self) -> dict:
        return {"score": self.score, "level": self.level, "n_features": self.n_features}


def _level_for(score: float | None) -> str:
    if score is None or np.isnan(score):
        return LEVEL_NORMAL
    if score >= HIGH_Z:
        return LEVEL_HIGH
    if score >= ELEVATED_Z:
        return LEVEL_ELEVATED
    return LEVEL_NORMAL


def compute_axis(name: str, z_scores: dict) -> AxisScore:
    """Aggregate one axis's features (signed) from a mapping of ``feature -> z-score``."""
    signs = AXIS_FEATURE_SIGNS[name]
    contributions = []
    for feature, sign in signs.items():
        z = z_scores.get(feature)
        if z is None or pd.isna(z):
            continue
        contributions.append(sign * float(z))
    if not contributions:
        return AxisScore(name=name, score=None, level=LEVEL_NORMAL, n_features=0)
    score = float(np.clip(np.mean(contributions), -SCORE_CLIP, SCORE_CLIP))
    return AxisScore(name=name, score=score, level=_level_for(score), n_features=len(contributions))


def compute_axes(z_scores: dict) -> dict[str, dict]:
    """Return all four axes as a JSON-ready dict from a mapping of ``feature -> z-score``.

    ``z_scores`` may use bare feature names (``hr_mean_bpm``) or the ``<feature>_z`` suffix that
    ``baseline.zscore_features`` produces; both are accepted.
    """
    normalized = {}
    for key, value in z_scores.items():
        base = key[:-2] if key.endswith("_z") else key
        normalized[base] = value
    return {name: compute_axis(name, normalized).to_dict() for name in AXIS_NAMES}
