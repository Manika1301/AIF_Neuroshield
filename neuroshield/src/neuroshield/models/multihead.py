"""D2: the multi-output model -- Head A (graded stress) + Head B (affect 4-class).

Head A is a calibrated gradient-boosting classifier trained on WESAD (Stress-Predict and Nurse are
held out -- see ``features.harmonize``). Calibration (isotonic) makes ``predict_proba`` a real
probability, so the 0--100 stress index is meaningful rather than a raw score. Head B is a
gradient-boosting 4-class classifier trained on WESAD's baseline/stress/amusement/meditation labels
-- it distinguishes stress-arousal from positive-arousal (amusement) and calm (meditation).

Both heads consume ``MODEL_FEATURE_COLUMNS``: the absolute features *plus* their per-subject
personal-baseline deviations (``features.personalize``). The personalized half is what lets the model
transfer to a person -- and a dataset -- it has never seen.

HistGradientBoosting handles NaN features natively, so no imputer is needed; a scaler is likewise
unnecessary for trees. Both heads are evaluated with grouped leave-one-subject-out against a dummy
baseline (see ``evaluate_head_a`` / ``evaluate_head_b``). The frozen artifact
(``m3_multihead_personalized_v1``) is versioned and never overwrites the validated M1 artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from neuroshield.features.extract import FEATURE_VERSION
from neuroshield.features.harmonize import AFFECT_CLASSES, training_view
from neuroshield.features.personalize import MODEL_FEATURE_COLUMNS
from neuroshield.models.artifact import (
    DEFAULT_THRESHOLD_POLICY,
    IncompatibleFeatureVersionError,
    ModelIntegrityError,
    _git_commit,
    _sha256_of_file,
)

MODEL_VERSION = "m3_multihead_personalized_v1"

DEFAULT_MODEL_PATH = Path("artifacts/models/m3_multihead_personalized_v1.joblib")
DEFAULT_MANIFEST_PATH = Path("artifacts/models/m3_multihead_personalized_v1_manifest.json")
DEFAULT_METRICS_PATH = Path("artifacts/metrics/m3_multihead_loso.json")

AFFECT_CODE_TO_NAME = {code: name for name, code in AFFECT_CLASSES.items()}

LEVEL_CALM = "calm"
LEVEL_ELEVATED = "elevated"
LEVEL_HIGH = "high"


def _base_gbm(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=4, l2_regularization=1.0, random_state=random_state
    )


def build_head_a(random_state: int = 0, calibration_cv: int = 3) -> CalibratedClassifierCV:
    """Calibrated binary stress classifier (isotonic over a gradient-boosting base)."""
    return CalibratedClassifierCV(
        estimator=_base_gbm(random_state), method="isotonic", cv=calibration_cv
    )


def build_head_b(random_state: int = 0) -> HistGradientBoostingClassifier:
    """4-class affect classifier."""
    return _base_gbm(random_state)


@dataclass
class MultiHeadModel:
    head_a: CalibratedClassifierCV
    head_b: HistGradientBoostingClassifier | None
    feature_columns: list[str]
    feature_version: str
    threshold_policy: dict

    def _level(self, prob: float) -> str:
        if prob < self.threshold_policy["green_max"]:
            return LEVEL_CALM
        if prob < self.threshold_policy["amber_max"]:
            return LEVEL_ELEVATED
        return LEVEL_HIGH

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return one row per input window: stress prob/index/level + affect state/confidence."""
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"input features are missing required columns: {missing}")
        X = features[self.feature_columns].to_numpy(dtype=float)

        stress_prob = self.head_a.predict_proba(X)[:, 1]
        out = pd.DataFrame(
            {
                "stress_prob": stress_prob,
                "stress_index": np.rint(stress_prob * 100).astype(int),
                "level": [self._level(p) for p in stress_prob],
            }
        )
        if self.head_b is not None:
            affect_proba = self.head_b.predict_proba(X)
            affect_code = self.head_b.classes_[np.argmax(affect_proba, axis=1)]
            out["affect_state"] = [AFFECT_CODE_TO_NAME.get(int(c), str(c)) for c in affect_code]
            out["affect_confidence"] = np.max(affect_proba, axis=1)
        else:
            out["affect_state"] = None
            out["affect_confidence"] = np.nan
        return out


def _fold_metrics_binary(y_true, y_pred) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_windows": int(len(y_true)),
    }


def evaluate_head_a(pooled: pd.DataFrame, random_state: int = 0) -> dict:
    """Grouped leave-one-subject-out for Head A over the train pool, vs. a dummy baseline."""
    train = training_view(pooled, "head_a")
    X = train[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train["head_a_label"].to_numpy(dtype=int)
    groups = train["group"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("Head A LOSO needs >= 2 groups")

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, y_dummy_all = [], [], []
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue  # a fold whose training rows are single-class can't calibrate; skip
        model = build_head_a(random_state)
        model.fit(X[tr], y[tr])
        y_pred_all.extend((model.predict_proba(X[te])[:, 1] >= 0.5).astype(int))
        dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
        y_dummy_all.extend(dummy.predict(X[te]))
        y_true_all.extend(y[te])

    return {
        "head": "head_a",
        "n_groups": int(len(np.unique(groups))),
        "n_windows": int(len(train)),
        "datasets": sorted(set(train["dataset"])),
        "model": _fold_metrics_binary(np.array(y_true_all), np.array(y_pred_all)),
        "dummy": _fold_metrics_binary(np.array(y_true_all), np.array(y_dummy_all)),
    }


def evaluate_head_b(pooled: pd.DataFrame, random_state: int = 0) -> dict:
    """Grouped leave-one-subject-out for Head B (WESAD affect), vs. a dummy baseline."""
    train = training_view(pooled, "head_b")
    labels = sorted(AFFECT_CLASSES.values())
    X = train[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train["head_b_label"].to_numpy(dtype=int)
    groups = train["group"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("Head B LOSO needs >= 2 groups")

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, y_dummy_all = [], [], []
    for tr, te in logo.split(X, y, groups):
        model = build_head_b(random_state).fit(X[tr], y[tr])
        y_pred_all.extend(model.predict(X[te]))
        dummy = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
        y_dummy_all.extend(dummy.predict(X[te]))
        y_true_all.extend(y[te])

    yt, yp, yd = np.array(y_true_all), np.array(y_pred_all), np.array(y_dummy_all)
    return {
        "head": "head_b",
        "classes": {name: code for name, code in AFFECT_CLASSES.items()},
        "n_groups": int(len(np.unique(groups))),
        "n_windows": int(len(train)),
        "model": {
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
            "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
        },
        "dummy": {
            "balanced_accuracy": float(balanced_accuracy_score(yt, yd)),
            "macro_f1": float(f1_score(yt, yd, average="macro", labels=labels, zero_division=0)),
        },
    }


def train_final_multihead(pooled: pd.DataFrame, random_state: int = 0) -> MultiHeadModel:
    """Fit both heads on all eligible rows (no held-out fold). This is the frozen model."""
    train_a = training_view(pooled, "head_a")
    head_a = build_head_a(random_state)
    head_a.fit(train_a[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float), train_a["head_a_label"].to_numpy(dtype=int))

    train_b = training_view(pooled, "head_b")
    head_b = None
    if len(train_b) > 0 and train_b["head_b_label"].nunique() > 1:
        head_b = build_head_b(random_state)
        head_b.fit(train_b[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float), train_b["head_b_label"].to_numpy(dtype=int))

    return MultiHeadModel(
        head_a=head_a,
        head_b=head_b,
        feature_columns=list(MODEL_FEATURE_COLUMNS),
        feature_version=FEATURE_VERSION,
        threshold_policy=dict(DEFAULT_THRESHOLD_POLICY),
    )


def save_multihead_artifact(
    model: MultiHeadModel,
    pooled: pd.DataFrame,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    train_a = training_view(pooled, "head_a")
    train_b = training_view(pooled, "head_b")
    manifest = {
        "model_version": MODEL_VERSION,
        "model_type": "multihead: calibrated_hgb (head_a) + hgb_4class (head_b)",
        "feature_version": model.feature_version,
        "feature_columns": list(model.feature_columns),
        "heads": {
            "head_a": {
                "task": "binary graded stress",
                "training_datasets": sorted(set(train_a["dataset"])),
                "n_windows": int(len(train_a)),
                "n_groups": int(train_a["group"].nunique()),
            },
            "head_b": {
                "task": "affect 4-class (baseline/stress/amusement/meditation)",
                "training_datasets": sorted(set(train_b["dataset"])),
                "n_windows": int(len(train_b)),
                "trained": model.head_b is not None,
            },
        },
        "threshold_policy": model.threshold_policy,
        "code_commit": _git_commit(),
        "metrics_path": str(metrics_path),
        "checksum_sha256": _sha256_of_file(model_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_multihead_artifact(
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    expected_feature_version: str = FEATURE_VERSION,
) -> tuple[MultiHeadModel, dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"multihead manifest not found at {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"multihead model not found at {model_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("feature_version") != expected_feature_version:
        raise IncompatibleFeatureVersionError(
            f"multihead manifest feature_version={manifest.get('feature_version')!r} != "
            f"expected {expected_feature_version!r}; refusing to load."
        )
    if _sha256_of_file(model_path) != manifest.get("checksum_sha256"):
        raise ModelIntegrityError(f"checksum mismatch for {model_path}; file may be corrupted.")
    return joblib.load(model_path), manifest


if __name__ == "__main__":
    raise SystemExit(
        "Do not run this module with `python -m` (it mis-pickles MultiHeadModel as __main__). "
        "Use: uv run python scripts/train_multihead.py"
    )
