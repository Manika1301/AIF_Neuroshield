"""Per-subject baseline personalization of the model's input features.

The single largest published lever on cross-subject wrist-based stress detection is not a fancier
classifier -- it is expressing each window *relative to that person's own quiet state* instead of in
absolute units. Two people can both be calm at 58 and 82 bpm; a model fed raw bpm has to waste
capacity learning who is who, and it simply cannot do that for a subject (or a whole dataset) it has
never seen. Schmidt et al. (WESAD, 2018), Siirtola (2019) and Gil-Martin et al. (2022) all report
subject-relative normalization as the biggest cross-subject gain, and it is also the honest fix for
our cross-dataset gap: WESAD, Stress-Predict and the nurse shifts differ in absolute offsets
(different rooms, different sensor seating, different populations) far more than they differ in
*deviations from a person's own baseline*.

The reference is each subject's **first few minutes of accepted windows** -- the same "sit quietly
for a few minutes" calibration the live app already performs (``runtime.baseline``). That makes this
deployable rather than a paper trick: nothing here peeks at labels, and nothing here peeks at a
window's own future. At inference the engine passes the real calibration profile
(``profile=...``) so the served features are computed exactly the way the trained ones were.

Personalized columns are *added alongside* the raw ones (suffix ``_p``), never replacing them: the
absolute level of a signal still carries information (very high EDA is unusual for anyone), so the
model sees both "how aroused is this person" and "how aroused is this person *for them*".
"""

from __future__ import annotations

import pandas as pd

from neuroshield.features.extract import FEATURE_COLUMNS
from neuroshield.features.labels import DEFAULT_MIN_VALID_FRACTION

# Quality/coverage metrics describe the *recording*, not the person's physiology -- personalizing
# them would be meaningless, so they are passed through raw only.
PERSONALIZE_EXCLUDE = ("ppg_quality", "valid_fraction")

PERSONALIZED_SUFFIX = "_p"
PERSONALIZE_BASE = [c for c in FEATURE_COLUMNS if c not in PERSONALIZE_EXCLUDE]
PERSONALIZED_COLUMNS = [f"{c}{PERSONALIZED_SUFFIX}" for c in PERSONALIZE_BASE]

# What the multi-head model actually consumes: absolute features + personal-baseline deviations.
MODEL_FEATURE_COLUMNS = list(FEATURE_COLUMNS) + PERSONALIZED_COLUMNS

# The reference period is defined in SECONDS of signal, not in a count of windows. A window count
# would silently mean different things at different window steps -- 10 windows is ~5 minutes at a
# 30s step but only ~2.5 minutes at a 10s step -- so the training reference would drift away from
# the fixed-duration calibration the live app performs. Seconds keep train and serve identical.
DEFAULT_REFERENCE_SECONDS = 300.0  # 5 minutes: stable, and short enough that a user will sit through it

# Below this, a "personal baseline" is noise pretending to be a reference.
MIN_REFERENCE_WINDOWS = 3

MIN_STD = 1e-3  # divide-by-zero floor, same convention as runtime.baseline


def _reference_stats(
    frame: pd.DataFrame, reference_seconds: float, min_valid_fraction: float
) -> tuple[pd.Series, pd.Series] | None:
    """Mean/std of one subject's accepted windows within the first ``reference_seconds``."""
    if "valid_fraction" in frame.columns:
        accepted = frame[frame["valid_fraction"] >= min_valid_fraction]
        if len(accepted) < MIN_REFERENCE_WINDOWS:
            accepted = frame  # too few clean windows: a noisy reference still beats none
    else:
        accepted = frame

    if "window_start_s" in accepted.columns:
        start = accepted["window_start_s"].min()
        reference = accepted[accepted["window_start_s"] < start + reference_seconds]
        if len(reference) < MIN_REFERENCE_WINDOWS:
            reference = accepted.head(MIN_REFERENCE_WINDOWS)
    else:
        reference = accepted.head(MIN_REFERENCE_WINDOWS)

    if reference.empty:
        return None
    means = reference[PERSONALIZE_BASE].mean(skipna=True)
    stds = reference[PERSONALIZE_BASE].std(skipna=True, ddof=0).clip(lower=MIN_STD)
    return means, stds


def _profile_stats(profile: dict) -> tuple[pd.Series, pd.Series]:
    means = pd.Series({c: profile["feature_means"][c] for c in PERSONALIZE_BASE})
    stds = pd.Series({c: profile["feature_stds"][c] for c in PERSONALIZE_BASE}).clip(lower=MIN_STD)
    return means, stds


def add_personalized_features(
    features: pd.DataFrame,
    profile: dict | None = None,
    subject_col: str = "subject_id",
    reference_seconds: float = DEFAULT_REFERENCE_SECONDS,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> pd.DataFrame:
    """Add a ``<feature>_p`` personal-baseline deviation column for every physiological feature.

    ``profile`` -- a ``runtime.baseline`` calibration profile -- is used as the reference when given
    (the live path). Otherwise the reference is derived per subject from that subject's own first
    ``reference_seconds`` of accepted windows (the training path). A subject with no usable window at
    all gets NaN, which the gradient-boosting heads handle natively.
    """
    missing = [c for c in PERSONALIZE_BASE if c not in features.columns]
    if missing:
        raise ValueError(f"cannot personalize: input is missing feature columns {missing}")

    result = features.copy()
    for col in PERSONALIZED_COLUMNS:
        result[col] = float("nan")

    if profile is not None:
        means, stds = _profile_stats(profile)
        for col in PERSONALIZE_BASE:
            result[f"{col}{PERSONALIZED_SUFFIX}"] = (result[col] - means[col]) / stds[col]
        return result

    if subject_col not in result.columns:
        raise ValueError(
            f"cannot personalize without a {subject_col!r} column (or an explicit calibration profile)"
        )

    sort_col = "window_start_s" if "window_start_s" in result.columns else None
    for _subject, idx in result.groupby(subject_col, sort=False).groups.items():
        frame = result.loc[idx]
        if sort_col:
            frame = frame.sort_values(sort_col)
        stats = _reference_stats(frame, reference_seconds, min_valid_fraction)
        if stats is None:
            continue
        means, stds = stats
        for col in PERSONALIZE_BASE:
            result.loc[idx, f"{col}{PERSONALIZED_SUFFIX}"] = (result.loc[idx, col] - means[col]) / stds[col]

    return result
