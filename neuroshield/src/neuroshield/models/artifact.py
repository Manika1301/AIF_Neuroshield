"""Save and load a versioned, manifest-checked M1 model artifact.

The frozen artifact used by the replay demo is trained on *all* eligible windows (not a LOSO
fold) -- its honest generalization estimate lives in the LOSO run's summary file (T8), referenced
from the manifest by path rather than recomputed here. Loading is intentionally strict: a feature
version mismatch or a corrupted file must fail loudly, never silently produce a prediction from
misaligned features.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuroshield.features.extract import FEATURE_COLUMNS, FEATURE_VERSION

MODEL_VERSION = "m1_wesad_features_v1"
MODEL_TYPE = "logistic_regression"

DEFAULT_MODEL_PATH = Path("artifacts/models/m1_wesad_features_v1.joblib")
DEFAULT_MANIFEST_PATH = Path("artifacts/models/m1_wesad_features_v1_manifest.json")
DEFAULT_METRICS_PATH = Path("artifacts/metrics/m1_loso_summary.json")

# Provisional status thresholds on P(stress); the runtime state machine (T14) is the source of
# truth once it exists, this is what the frozen artifact was calibrated/documented against.
DEFAULT_THRESHOLD_POLICY = {"green_max": 0.45, "amber_max": 0.70}


class IncompatibleFeatureVersionError(ValueError):
    pass


class ModelIntegrityError(ValueError):
    pass


class MissingFeatureColumnsError(ValueError):
    pass


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def train_final_m1(
    kept: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    random_state: int = 0,
):
    """Fit the pipeline on every eligible window -- no held-out subject, this is the frozen model."""
    X = kept[feature_columns].to_numpy(dtype=float)
    y = kept["m1_label"].to_numpy(dtype=int)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
    )
    pipeline.fit(X, y)
    return pipeline


def save_model_artifact(
    pipeline,
    kept: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    training_dataset: str = "wesad",
    metrics_path: Path = DEFAULT_METRICS_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    threshold_policy: dict = None,
) -> dict:
    """Save ``pipeline`` and its manifest. Returns the manifest dict that was written."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    checksum = _sha256_of_file(model_path)

    manifest = {
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "feature_version": FEATURE_VERSION,
        "feature_columns": list(feature_columns),
        "training_dataset": training_dataset,
        "n_subjects": int(kept["subject_id"].nunique()),
        "n_windows": int(len(kept)),
        "class_balance": {
            "baseline": int((kept["m1_label"] == 0).sum()),
            "stress": int((kept["m1_label"] == 1).sum()),
        },
        "code_commit": _git_commit(),
        "metrics_path": str(metrics_path),
        "threshold_policy": threshold_policy or DEFAULT_THRESHOLD_POLICY,
        "checksum_sha256": checksum,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_model_artifact(
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    expected_feature_version: str = FEATURE_VERSION,
) -> tuple[object, dict]:
    """Load the model, refusing to return anything for an incompatible or corrupted artifact."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"model manifest not found at {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found at {model_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest.get("feature_version") != expected_feature_version:
        raise IncompatibleFeatureVersionError(
            f"model manifest declares feature_version={manifest.get('feature_version')!r}, "
            f"but the running code expects {expected_feature_version!r}. Refusing to load: "
            "predictions from mismatched feature versions would be silently wrong."
        )

    actual_checksum = _sha256_of_file(model_path)
    if actual_checksum != manifest.get("checksum_sha256"):
        raise ModelIntegrityError(
            f"checksum mismatch for {model_path}: manifest says "
            f"{manifest.get('checksum_sha256')!r}, file hashes to {actual_checksum!r}. "
            "The model file may be corrupted or does not match this manifest."
        )

    pipeline = joblib.load(model_path)
    return pipeline, manifest


def predict_proba_stress(pipeline, manifest: dict, features: pd.DataFrame):
    """Predict P(stress) for ``features``, enforcing the manifest's exact feature column order."""
    required = manifest["feature_columns"]
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise MissingFeatureColumnsError(
            f"input features are missing required columns: {missing}"
        )
    X = features[required].to_numpy(dtype=float)
    return pipeline.predict_proba(X)[:, 1]


if __name__ == "__main__":
    import pandas as pd

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

    pipeline = train_final_m1(kept)
    manifest = save_model_artifact(pipeline, kept)
    print(f"saved {DEFAULT_MODEL_PATH} (checksum {manifest['checksum_sha256'][:12]}...)")
    print(f"saved {DEFAULT_MANIFEST_PATH}")

    reloaded_pipeline, reloaded_manifest = load_model_artifact()
    print(f"reloaded OK: model_version={reloaded_manifest['model_version']} "
          f"feature_version={reloaded_manifest['feature_version']}")
