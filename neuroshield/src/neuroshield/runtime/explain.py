"""Plain-language explanations: turn ranked feature deviations into restrained reason sentences.

Every colored (green/amber/red) status gets 1-3 ranked reasons. Ranking uses signed
contributions (LR coefficient x personal z-score) when a fitted LogisticRegression is available,
falling back to plain |z-score| ranking otherwise (T15 step 3, for a future non-LR model). The
*direction* worded in each sentence ("above"/"below your quiet baseline") always reflects the raw
personal z-score, not the contribution sign -- a sentence is a factual statement about this
person's own signal, not a claim about what drove the model's decision.

Explanations never diagnose. See docs/no_clinical_claims.md; ``assert_no_clinical_claims``
below is a regression guard against ever emitting language like "panic attack starting".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from neuroshield.runtime.quality_gate import MOTION_PAUSED, POOR_SIGNAL, AbstentionResult
from neuroshield.runtime.status import AMBER, GREEN, RED

# Only features with an established, restrained physiological phrasing are eligible to be
# surfaced as a reason. Quality/coverage metrics (ppg_quality, valid_fraction) are excluded --
# they gate abstention (T13), they are not a physiological explanation for a colored status.
FEATURE_GROUPS = {
    "hr_mean_bpm": "pulse",
    "ibi_sd_ms": "pulse",
    "ibi_rmssd_ms": "pulse",
    "hrv_lf_hf_ratio": "pulse",
    "eda_level": "eda",
    "eda_slope": "eda",
    "eda_response_count": "eda",
    "eda_response_mean_amp": "eda",
    "eda_tonic_mean": "eda",
    "eda_phasic_mean": "eda",
    "temp_mean_c": "temperature",
    "temp_slope_c_per_min": "temperature",
}

TEMPLATES = {
    "eda": {
        "above": "Skin-response activity is above your quiet baseline.",
        "below": "Skin-response activity is below your quiet baseline.",
    },
    "pulse": {
        "above": "Pulse features are higher than your quiet baseline.",
        "below": "Pulse features are lower than your quiet baseline.",
    },
    "temperature": {
        "above": "Temperature trend is higher than your quiet baseline.",
        "below": "Temperature trend is lower than your quiet baseline.",
    },
}

EXPLAINABLE_FEATURES = list(FEATURE_GROUPS.keys())

MOTION_PAUSED_REASON = "Hand motion is high, so the model is paused."
POOR_SIGNAL_REASON = "Signal quality is too low right now, so the model is paused."
FALLBACK_REASON = "No single feature stood out; the overall pattern differs from your quiet baseline."

MAX_REASONS = 3

# Regression guard for docs/no_clinical_claims.md: none of these may ever appear in output text.
FORBIDDEN_TERMS = [
    "panic",
    "attack",
    "diagnos",
    "burnout",
    "anxiety",
    "heat illness",
    "medical",
    "clinical",
    "disorder",
]


@dataclass
class Reason:
    feature: str
    group: str
    direction: str  # "above" | "below"
    z_score: float
    contribution: float | None
    text: str


def assert_no_clinical_claims(text: str) -> None:
    lowered = text.lower()
    for term in FORBIDDEN_TERMS:
        if term in lowered:
            raise ValueError(f"explanation text contains a forbidden clinical term {term!r}: {text!r}")


def extract_lr_coefficients(pipeline, feature_columns: list[str]) -> dict[str, float]:
    """Pull the fitted LogisticRegression step's coefficients, aligned to feature_columns order."""
    model = pipeline[-1]
    return dict(zip(feature_columns, model.coef_[0]))


def _direction(z: float) -> str:
    return "above" if z >= 0 else "below"


def rank_reasons(
    z_scores: dict[str, float],
    coefficients: dict[str, float] | None = None,
    max_reasons: int = MAX_REASONS,
) -> list[Reason]:
    """Rank explainable features and return up to ``max_reasons``, at most one per group."""
    candidates = []
    for feature, group in FEATURE_GROUPS.items():
        z = z_scores.get(feature)
        if z is None or pd.isna(z):
            continue
        if coefficients is not None:
            contribution = coefficients.get(feature, 0.0) * z
            score = abs(contribution)
        else:
            contribution = None
            score = abs(z)
        candidates.append((score, feature, group, z, contribution))

    candidates.sort(key=lambda c: c[0], reverse=True)

    reasons: list[Reason] = []
    used_groups: set[str] = set()
    for _score, feature, group, z, contribution in candidates:
        if group in used_groups:
            continue
        direction = _direction(z)
        text = TEMPLATES[group][direction]
        assert_no_clinical_claims(text)
        reasons.append(
            Reason(feature=feature, group=group, direction=direction, z_score=z, contribution=contribution, text=text)
        )
        used_groups.add(group)
        if len(reasons) >= max_reasons:
            break

    return reasons


def explain_color_status(
    z_scores: dict[str, float],
    coefficients: dict[str, float] | None = None,
    max_reasons: int = MAX_REASONS,
) -> list[str]:
    """Reason text for a green/amber/red status. Always returns at least one reason."""
    reasons = rank_reasons(z_scores, coefficients, max_reasons=max_reasons)
    if not reasons:
        assert_no_clinical_claims(FALLBACK_REASON)
        return [FALLBACK_REASON]
    return [r.text for r in reasons]


def explain_abstention(abstention: AbstentionResult) -> list[str]:
    if abstention.reason == MOTION_PAUSED:
        text = MOTION_PAUSED_REASON
    elif abstention.reason == POOR_SIGNAL:
        text = POOR_SIGNAL_REASON
    else:
        raise ValueError(f"unknown abstention reason: {abstention.reason!r}")
    assert_no_clinical_claims(text)
    return [text]


def explain_status(
    state: str,
    z_scores: dict[str, float] | None = None,
    coefficients: dict[str, float] | None = None,
    abstention: AbstentionResult | None = None,
    max_reasons: int = MAX_REASONS,
) -> list[str]:
    """Top-level entry point: reason text appropriate for whatever state the status machine is in."""
    if state in (MOTION_PAUSED, POOR_SIGNAL):
        if abstention is None:
            raise ValueError(f"state={state!r} requires an AbstentionResult to explain")
        return explain_abstention(abstention)
    if state in (GREEN, AMBER, RED):
        if z_scores is None:
            raise ValueError(f"state={state!r} requires z_scores to explain")
        return explain_color_status(z_scores, coefficients, max_reasons=max_reasons)
    return []  # waiting / calibrating / stale / error: nothing physiological to explain yet
