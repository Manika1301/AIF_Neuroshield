"""Personal baseline profiles: calibrate feature z-scoring from a quiet segment of a session.

A baseline profile records each feature's mean and standard deviation over a quiet/rest period,
so live features can be expressed relative to *this person's* quiet state rather than a
population average -- the personalization step in the software-first runtime.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from neuroshield.features.extract import FEATURE_COLUMNS, FEATURE_VERSION, extract_features
from neuroshield.features.labels import DEFAULT_MIN_VALID_FRACTION
from neuroshield.models.artifact import IncompatibleFeatureVersionError, MissingFeatureColumnsError
from neuroshield.runtime.events_to_bundle import events_to_bundle

DEFAULT_MIN_STD = 1e-3  # floor applied to every feature's std to guard against divide-by-zero

DEFAULT_BASELINE_PATH = Path("artifacts/baselines/live_baseline.json")


def compute_baseline_profile(
    quiet_features: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    source: str = "unknown",
    subject_id: str = "live",
    min_std: float = DEFAULT_MIN_STD,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> dict:
    """Build a baseline profile dict from windowed features of a quiet segment.

    Only windows meeting ``min_valid_fraction`` coverage are "accepted" into the calibration --
    a baseline built from low-quality windows would misrepresent the person's true quiet state.
    """
    if "valid_fraction" in quiet_features.columns:
        accepted = quiet_features[quiet_features["valid_fraction"] >= min_valid_fraction]
    else:
        accepted = quiet_features

    if len(accepted) == 0:
        raise ValueError(
            "compute_baseline_profile: no windows met the valid_fraction threshold "
            f"({min_valid_fraction}); cannot calibrate a baseline from zero accepted windows"
        )

    means = accepted[feature_columns].mean(skipna=True)
    stds = accepted[feature_columns].std(skipna=True, ddof=0).clip(lower=min_std)

    if "window_start_s" in accepted.columns and "window_end_s" in accepted.columns:
        accepted_seconds = float(accepted["window_end_s"].max() - accepted["window_start_s"].min())
    else:
        accepted_seconds = float(len(accepted))

    return {
        "feature_version": FEATURE_VERSION,
        "feature_means": {k: float(v) for k, v in means.items()},
        "feature_stds": {k: float(v) for k, v in stds.items()},
        "accepted_seconds": accepted_seconds,
        "n_accepted_windows": int(len(accepted)),
        "n_total_windows": int(len(quiet_features)),
        "min_std_floor": min_std,
        "source": source,
        "subject_id": subject_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_baseline_from_events(
    quiet_events: list[dict],
    source: str = "unknown",
    subject_id: str = "live",
    window_sec: float = 60.0,
    step_sec: float = 30.0,
    min_std: float = DEFAULT_MIN_STD,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> dict:
    """Compute a baseline profile directly from a quiet segment's raw contract events."""
    bundle = events_to_bundle(quiet_events, dataset=source, subject_id=subject_id)
    quiet_features = extract_features(bundle, window_sec=window_sec, step_sec=step_sec)
    return compute_baseline_profile(
        quiet_features,
        source=source,
        subject_id=subject_id,
        min_std=min_std,
        min_valid_fraction=min_valid_fraction,
    )


def save_baseline_profile(profile: dict, path: Path = DEFAULT_BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


def load_baseline_profile(path: Path = DEFAULT_BASELINE_PATH, expected_feature_version: str = FEATURE_VERSION) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"baseline profile not found at {path}")

    with open(path) as f:
        profile = json.load(f)

    if profile.get("feature_version") != expected_feature_version:
        raise IncompatibleFeatureVersionError(
            f"baseline profile declares feature_version={profile.get('feature_version')!r}, "
            f"but the running code expects {expected_feature_version!r}. Refusing to load: "
            "z-scoring against a mismatched baseline would silently misalign features."
        )

    return profile


def zscore_features(
    features: pd.DataFrame, profile: dict, feature_columns: list[str] = None
) -> pd.DataFrame:
    """Return a copy of ``features`` with a ``<col>_z`` column for every feature in the profile."""
    feature_columns = feature_columns or list(profile["feature_means"].keys())
    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        raise MissingFeatureColumnsError(f"input features are missing required columns: {missing}")

    means = pd.Series(profile["feature_means"])
    stds = pd.Series(profile["feature_stds"])

    result = features.copy()
    for col in feature_columns:
        result[f"{col}_z"] = (features[col] - means[col]) / stds[col]
    return result


if __name__ == "__main__":
    import argparse

    from neuroshield.runtime.replay_source import ReplaySource

    parser = argparse.ArgumentParser(description="Compute a baseline profile from a quiet replay segment")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--quiet-seconds", type=float, default=150.0, help="use events with t_us < this many seconds")
    parser.add_argument("--session-id", type=str, default=None, help="only use events from this session_id")
    parser.add_argument("--subject-id", type=str, default="live")
    parser.add_argument("--out", type=Path, default=DEFAULT_BASELINE_PATH)
    args = parser.parse_args()

    source = ReplaySource(args.replay, speed=None)
    quiet_events = []
    for event in source:
        if args.session_id and event["session_id"] != args.session_id:
            continue
        if event["t_us"] >= args.quiet_seconds * 1_000_000:
            break
        quiet_events.append(event)

    profile = compute_baseline_from_events(quiet_events, source=str(args.replay), subject_id=args.subject_id)
    save_baseline_profile(profile, args.out)
    print(f"wrote baseline profile ({profile['n_accepted_windows']} accepted windows, "
          f"{profile['accepted_seconds']:.0f}s) -> {args.out}")
