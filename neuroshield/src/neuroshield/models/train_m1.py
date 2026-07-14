"""Train and evaluate M1 (baseline vs stress-proxy) with leave-one-subject-out cross-validation.

LeaveOneGroupOut, grouped by ``subject_id``, guarantees that a subject held out as the test fold
never contributes any window to that fold's training data -- the only way to get an honest read
on how M1 generalizes to a person it has never seen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuroshield.features.extract import FEATURE_COLUMNS

MODEL_TYPE = "logistic_regression"

DEFAULT_PREDICTIONS_PATH = Path("artifacts/metrics/m1_loso_predictions.csv")
DEFAULT_SUMMARY_PATH = Path("artifacts/metrics/m1_loso_summary.json")


def _build_pipeline(random_state: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
    )


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_windows": int(len(y_true)),
        "n_baseline": int(np.sum(y_true == 0)),
        "n_stress": int(np.sum(y_true == 1)),
    }


def train_and_evaluate_m1(
    kept: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    random_state: int = 0,
) -> dict:
    """Run LOSO cross-validation for M1 and a dummy baseline over ``kept`` windows.

    ``kept`` must be the output of ``label_m1_binary`` (has ``subject_id``, ``m1_label``, and
    every column in ``feature_columns``). Returns a dict with ``"predictions"`` (a DataFrame, one
    row per window, with both models' predictions) and ``"summary"`` (JSON-serializable metrics:
    overall model/dummy comparison plus per-subject model metrics).
    """
    X = kept[feature_columns].to_numpy(dtype=float)
    y = kept["m1_label"].to_numpy(dtype=int)
    groups = kept["subject_id"].to_numpy()
    subject_ids = kept["subject_id"].to_numpy()

    if len(np.unique(groups)) < 2:
        raise ValueError("LeaveOneGroupOut requires at least 2 subjects; got fewer")

    logo = LeaveOneGroupOut()
    rows = []
    for train_idx, test_idx in logo.split(X, y, groups):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        if not train_subjects.isdisjoint(test_subjects):
            raise AssertionError("a subject's windows appeared in both the train and test fold")

        pipeline = _build_pipeline(random_state)
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])
        y_prob = pipeline.predict_proba(X[test_idx])[:, 1]

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X[train_idx], y[train_idx])
        y_pred_dummy = dummy.predict(X[test_idx])

        for pos, idx in enumerate(test_idx):
            rows.append(
                {
                    "subject_id": subject_ids[idx],
                    "y_true": int(y[idx]),
                    "y_pred_model": int(y_pred[pos]),
                    "y_prob_model": float(y_prob[pos]),
                    "y_pred_dummy": int(y_pred_dummy[pos]),
                }
            )

    predictions = pd.DataFrame(rows)

    overall_model = _fold_metrics(predictions["y_true"].to_numpy(), predictions["y_pred_model"].to_numpy())
    overall_dummy = _fold_metrics(predictions["y_true"].to_numpy(), predictions["y_pred_dummy"].to_numpy())

    per_subject = {}
    for subject_id, group_df in predictions.groupby("subject_id"):
        per_subject[str(subject_id)] = _fold_metrics(
            group_df["y_true"].to_numpy(), group_df["y_pred_model"].to_numpy()
        )

    summary = {
        "model_type": MODEL_TYPE,
        "feature_columns": list(feature_columns),
        "n_subjects": int(len(np.unique(groups))),
        "n_windows": int(len(kept)),
        "class_balance": {
            "baseline": int(np.sum(y == 0)),
            "stress": int(np.sum(y == 1)),
        },
        "overall": {
            "model": overall_model,
            "dummy": overall_dummy,
        },
        "per_subject": per_subject,
    }

    return {"predictions": predictions, "summary": summary}


def save_results(
    result: dict,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(predictions_path, index=False)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(result["summary"], f, indent=2)


if __name__ == "__main__":
    from neuroshield.data.bundle import wesad_subject_to_bundle
    from neuroshield.data.wesad_loader import load_wesad_subject
    from neuroshield.features.extract import extract_features
    from neuroshield.features.labels import label_m1_binary

    subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]
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
    kept, _ = label_m1_binary(all_windows)

    result = train_and_evaluate_m1(kept)
    save_results(result)

    summary = result["summary"]
    print(f"subjects={summary['n_subjects']} windows={summary['n_windows']} "
          f"class_balance={summary['class_balance']}")
    print(f"model:  balanced_accuracy={summary['overall']['model']['balanced_accuracy']:.3f} "
          f"macro_f1={summary['overall']['model']['macro_f1']:.3f}")
    print(f"dummy:  balanced_accuracy={summary['overall']['dummy']['balanced_accuracy']:.3f} "
          f"macro_f1={summary['overall']['dummy']['macro_f1']:.3f}")
