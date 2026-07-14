"""Turn raw WESAD window labels (from extract_features) into clean M1 baseline-vs-stress labels.

M1 is a binary classifier: baseline (0) vs stress (1). Everything else in the WESAD protocol
(amusement, meditation, transient/undefined periods, and the unused label codes 5-7) is excluded
from M1 training, not coerced into one of the two classes. Windows that do carry a baseline or
stress label but do not have enough valid signal coverage are dropped separately, so the two
exclusion reasons stay distinguishable in the count table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuroshield.data.wesad_loader import LABEL_NAMES

# WESAD raw label codes that map onto the M1 binary task.
BASELINE_RAW_LABEL = 1
STRESS_RAW_LABEL = 2
RAW_TO_BINARY = {BASELINE_RAW_LABEL: 0, STRESS_RAW_LABEL: 1}

DEFAULT_MIN_VALID_FRACTION = 0.9

DEFAULT_LABEL_COUNTS_PATH = Path("artifacts/metrics/wesad_label_counts.csv")


def _exclusion_reason(raw_label: float, valid_fraction: float, min_valid_fraction: float) -> str | None:
    if pd.isna(raw_label) or int(raw_label) not in RAW_TO_BINARY:
        raw_name = LABEL_NAMES.get(int(raw_label), f"unknown_{raw_label}") if pd.notna(raw_label) else "missing"
        return f"non_binary_label:{raw_name}"
    if pd.isna(valid_fraction) or valid_fraction < min_valid_fraction:
        return "excluded_low_valid_fraction"
    return None


def label_m1_binary(
    windows: pd.DataFrame, min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate every window with its M1 fate, and return (kept_rows, label_count_table).

    ``windows`` must have ``subject_id``, ``label`` (raw WESAD code), and ``valid_fraction``
    columns, e.g. the concatenated output of ``extract_features`` across subjects.

    Returns:
        kept: a copy of ``windows`` restricted to rows with exactly one binary label, with a new
            ``m1_label`` int column (0 = baseline, 1 = stress). Every other input column,
            including ``subject_id``, is preserved unchanged.
        counts: a long-format table with one row per (subject_id, category, count), where
            category is ``"baseline"``, ``"stress"``, or an ``"excluded_*"`` reason. Includes an
            ``"ALL"`` subject_id aggregate. This is what gets saved to
            ``artifacts/metrics/wesad_label_counts.csv``.
    """
    annotated = windows.copy()
    annotated["exclusion_reason"] = [
        _exclusion_reason(row.label, row.valid_fraction, min_valid_fraction)
        for row in annotated.itertuples()
    ]

    kept_mask = annotated["exclusion_reason"].isna()
    kept = annotated.loc[kept_mask].copy()
    kept["m1_label"] = kept["label"].astype(int).map(RAW_TO_BINARY)
    assert kept["m1_label"].isin([0, 1]).all(), "every kept row must have exactly one binary label"

    category = annotated["exclusion_reason"].copy()
    category.loc[kept_mask] = kept["m1_label"].map({0: "baseline", 1: "stress"})

    counts_per_subject = (
        annotated.assign(category=category)
        .groupby(["subject_id", "category"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    counts_total = (
        counts_per_subject.groupby("category")["count"]
        .sum()
        .reset_index()
        .assign(subject_id="ALL")
    )
    counts = pd.concat([counts_per_subject, counts_total], ignore_index=True)
    counts = counts.sort_values(["subject_id", "category"]).reset_index(drop=True)

    return kept, counts


def save_label_counts(counts: pd.DataFrame, path: Path = DEFAULT_LABEL_COUNTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(path, index=False)


if __name__ == "__main__":
    import sys

    from neuroshield.data.bundle import wesad_subject_to_bundle
    from neuroshield.data.wesad_loader import load_wesad_subject
    from neuroshield.features.extract import extract_features

    subject_ids = sys.argv[1:] or [f"S{i}" for i in range(2, 18) if i != 12]

    frames = []
    for sid in subject_ids:
        try:
            subject = load_wesad_subject(sid)
        except FileNotFoundError as e:
            print(f"skipping {sid}: {e}")
            continue
        bundle = wesad_subject_to_bundle(subject)
        frames.append(extract_features(bundle))

    if not frames:
        raise SystemExit("no WESAD subjects available -- download WESAD first (see docs/datasets.md)")

    all_windows = pd.concat(frames, ignore_index=True)
    kept, counts = label_m1_binary(all_windows)
    save_label_counts(counts)
    print(counts.to_string(index=False))
    print(f"\nkept {len(kept)} of {len(all_windows)} windows for M1 training")
