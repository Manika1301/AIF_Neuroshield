"""D1: harmonize labels across WESAD, Stress-Predict, and Nurse Stress into a common schema.

The three datasets carry different raw label spaces (WESAD's protocol codes, Stress-Predict's
per-second 0/1, Nurse's survey-interval 0/1). This module maps all of them onto the two model
heads of the redesign (see docs/design_doc.tex) and records exactly which windows are excluded and
why -- so a pooled training/eval table is reproducible and the exclusion accounting is auditable.

Two heads:
  - Head A (graded stress, binary): baseline -> 0, stress -> 1. Trained on the pooled
    train-pool datasets (WESAD + Stress-Predict); Nurse is held out.
  - Head B (affect, 4-class): baseline/stress/amusement/meditation -> 0/1/2/3. Only WESAD carries
    these labels, so only WESAD windows are ever eligible for Head B.

Group keys are ``"<dataset>:<participant>"`` so grouped cross-validation never splits one person
across train/eval. For Nurse, the per-session subject_id (``"<ID>_<unix>"``) collapses to the
participant ``<ID>`` -- one nurse's many sessions form a single group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuroshield.features.labels import DEFAULT_MIN_VALID_FRACTION

# Head A (binary stress)
HEAD_A_BASELINE = 0
HEAD_A_STRESS = 1

# Head B (affect 4-class)
AFFECT_CLASSES = {"baseline": 0, "stress": 1, "amusement": 2, "meditation": 3}

# WESAD raw protocol codes (see neuroshield.data.wesad_loader.LABEL_NAMES)
_WESAD_HEAD_A = {1: HEAD_A_BASELINE, 2: HEAD_A_STRESS}
_WESAD_HEAD_B = {1: 0, 2: 1, 3: 2, 4: 3}  # baseline/stress/amusement/meditation
# Stress-Predict and Nurse loaders already emit binary 0/1 (or -1 for unlabeled)
_BINARY_HEAD_A = {0: HEAD_A_BASELINE, 1: HEAD_A_STRESS}

# Head A trains on WESAD only. Naive pooling with Stress-Predict was found to *lower* LOSO
# accuracy (0.83 -> 0.62) because it folds cross-protocol generalization into the CV without
# domain adaptation -- the wearable-stress literature reports the same effect. Stress-Predict and
# Nurse are therefore held out as honest external validation, not training signal.
TRAIN_POOL_DATASETS = ("wesad",)
HELDOUT_DATASETS = ("stress_predict", "nurse_stress")
KNOWN_DATASETS = TRAIN_POOL_DATASETS + HELDOUT_DATASETS

# Columns harmonize adds to a features table.
HARMONIZED_COLUMNS = [
    "dataset",
    "group",
    "split",
    "head_a_label",
    "head_b_label",
    "quality_ok",
    "eligible_head_a",
    "eligible_head_b",
]


def participant_group(dataset: str, subject_id: str) -> str:
    """Group key for CV: one physical person, regardless of how many sessions they have."""
    if dataset == "nurse_stress":
        participant = str(subject_id).split("_")[0]
    else:
        participant = str(subject_id)
    return f"{dataset}:{participant}"


def _head_a_label(dataset: str, raw_label) -> float:
    if pd.isna(raw_label):
        return np.nan
    code = int(raw_label)
    if dataset == "wesad":
        return _WESAD_HEAD_A.get(code, np.nan)
    if dataset in ("stress_predict", "nurse_stress"):
        return _BINARY_HEAD_A.get(code, np.nan)
    return np.nan


def _head_b_label(dataset: str, raw_label) -> float:
    if dataset != "wesad" or pd.isna(raw_label):
        return np.nan
    return _WESAD_HEAD_B.get(int(raw_label), np.nan)


def harmonize_labels(
    features: pd.DataFrame, dataset: str, min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION
) -> tuple[pd.DataFrame, dict]:
    """Annotate a single dataset's feature table with harmonized labels + eligibility.

    ``features`` is the output of ``extract_features`` for one dataset (has ``subject_id``,
    ``label``, ``valid_fraction``). Returns ``(harmonized_df, counts)`` where ``counts`` reports,
    for this dataset, how many windows are eligible for each head and how many were excluded and
    why (low quality vs. no usable label).
    """
    if dataset not in KNOWN_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {KNOWN_DATASETS}")

    df = features.copy()
    df["dataset"] = dataset
    df["group"] = [participant_group(dataset, s) for s in df["subject_id"]]
    df["split"] = "train_pool" if dataset in TRAIN_POOL_DATASETS else "heldout"
    df["head_a_label"] = [_head_a_label(dataset, v) for v in df["label"]]
    df["head_b_label"] = [_head_b_label(dataset, v) for v in df["label"]]

    quality_ok = df["valid_fraction"] >= min_valid_fraction if "valid_fraction" in df else True
    df["quality_ok"] = quality_ok
    df["eligible_head_a"] = df["head_a_label"].notna() & df["quality_ok"]
    df["eligible_head_b"] = df["head_b_label"].notna() & df["quality_ok"]

    counts = {
        "dataset": dataset,
        "split": "train_pool" if dataset in TRAIN_POOL_DATASETS else "heldout",
        "n_windows": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "head_a": {
            "eligible": int(df["eligible_head_a"].sum()),
            "baseline": int((df.loc[df["eligible_head_a"], "head_a_label"] == HEAD_A_BASELINE).sum()),
            "stress": int((df.loc[df["eligible_head_a"], "head_a_label"] == HEAD_A_STRESS).sum()),
            "excluded_no_label": int(df["head_a_label"].isna().sum()),
            "excluded_low_quality": int((df["head_a_label"].notna() & ~df["quality_ok"]).sum()),
        },
        "head_b": {
            "eligible": int(df["eligible_head_b"].sum()),
            "per_class": {
                name: int((df.loc[df["eligible_head_b"], "head_b_label"] == code).sum())
                for name, code in AFFECT_CLASSES.items()
            },
            "excluded_no_label": int(df["head_b_label"].isna().sum()),
            "excluded_low_quality": int((df["head_b_label"].notna() & ~df["quality_ok"]).sum()),
        },
    }
    return df, counts


def pool_harmonized(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-dataset harmonized tables into one pooled table with a global row index."""
    return pd.concat(frames, ignore_index=True)


def training_view(pooled: pd.DataFrame, head: str = "head_a") -> pd.DataFrame:
    """Rows eligible to *train* a head: train-pool split only, eligible for that head.

    Head A trains on WESAD + Stress-Predict; Head B trains on WESAD only (the only affect labels).
    Nurse rows are never returned here -- they are held out for evaluation by construction.
    """
    if head not in ("head_a", "head_b"):
        raise ValueError(f"head must be 'head_a' or 'head_b', got {head!r}")
    eligible_col = f"eligible_{head}"
    return pooled[(pooled["split"] == "train_pool") & pooled[eligible_col]].copy()


if __name__ == "__main__":
    import json

    from neuroshield.data.bundle import (
        nurse_stress_session_to_bundle,
        stress_predict_subject_to_bundle,
        wesad_subject_to_bundle,
    )
    from neuroshield.data.nurse_stress_loader import (
        list_participant_sessions,
        list_participants,
        load_nurse_stress_session,
        load_survey_events,
    )
    from neuroshield.data.stress_predict_loader import load_stress_predict_subject
    from neuroshield.data.wesad_loader import load_wesad_subject
    from neuroshield.features.extract import extract_features

    all_counts = []
    frames = []

    # WESAD
    for i in range(2, 18):
        if i == 12:
            continue
        try:
            bundle = wesad_subject_to_bundle(load_wesad_subject(f"S{i}"))
        except FileNotFoundError:
            continue
        h, c = harmonize_labels(extract_features(bundle), "wesad")
        frames.append(h)
        all_counts.append(c)

    # Stress-Predict
    for i in range(1, 36):
        try:
            bundle = stress_predict_subject_to_bundle(load_stress_predict_subject(f"S{i:02d}"))
        except FileNotFoundError:
            continue
        h, c = harmonize_labels(extract_features(bundle), "stress_predict")
        frames.append(h)
        all_counts.append(c)

    # Nurse Stress (per session)
    try:
        events = load_survey_events()
        for pid in list_participants():
            for sid in list_participant_sessions(pid):
                session = load_nurse_stress_session(sid, events=events)
                bundle = nurse_stress_session_to_bundle(session)
                h, c = harmonize_labels(extract_features(bundle), "nurse_stress")
                frames.append(h)
    except FileNotFoundError:
        pass

    if frames:
        pooled = pool_harmonized(frames)
        print(f"pooled rows: {len(pooled)} | groups: {pooled['group'].nunique()}")
        print(f"Head A train rows: {len(training_view(pooled, 'head_a'))}")
        print(f"Head B train rows: {len(training_view(pooled, 'head_b'))}")
        print(json.dumps(all_counts[:3], indent=2))
