"""D3: one comparable, honest scoreboard of Head A performance across all three datasets.

Honesty subtlety: WESAD and Stress-Predict are both *in* Head A's training pool, so asking the
frozen model to predict on them would be train-on-test and inflate the numbers. Instead, the
train-pool datasets are scored from **grouped leave-one-subject-out** predictions bucketed per
dataset (when a WESAD subject is held out, its predictions count toward WESAD; likewise for
Stress-Predict). Nurse Stress is genuinely held out, so the frozen model is run on it directly.

The result is three comparable rows -- generalization to a new WESAD person, to Stress-Predict's
different lab protocol, and to real hospital shifts -- each an honest out-of-training estimate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from neuroshield.features.harmonize import training_view
from neuroshield.features.personalize import MODEL_FEATURE_COLUMNS
from neuroshield.models.multihead import build_head_a

DEFAULT_SCOREBOARD_JSON = Path("artifacts/metrics/validation_scoreboard.json")
DEFAULT_SCOREBOARD_MD = Path("artifacts/metrics/validation_scoreboard.md")


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_windows": int(len(y_true)),
    }


def loso_predictions_by_dataset(pooled: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """Grouped-LOSO over the train pool, returning one row per held-out window with its dataset."""
    train = training_view(pooled, "head_a")
    X = train[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train["head_a_label"].to_numpy(dtype=int)
    groups = train["group"].to_numpy()
    datasets = train["dataset"].to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("scoreboard LOSO needs >= 2 groups")

    rows = []
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        model = build_head_a(random_state)
        model.fit(X[tr], y[tr])
        preds = (model.predict_proba(X[te])[:, 1] >= 0.5).astype(int)
        for pos, idx in enumerate(te):
            rows.append({"dataset": datasets[idx], "y_true": int(y[idx]), "y_pred": int(preds[pos])})
    return pd.DataFrame(rows)


def heldout_predictions(model, heldout_pooled: pd.DataFrame) -> pd.DataFrame:
    """Run the frozen model on a held-out dataset's eligible Head A windows."""
    eligible = heldout_pooled[heldout_pooled["eligible_head_a"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["dataset", "y_true", "y_pred"])
    pred = model.predict(eligible)
    y_pred = (pred["stress_prob"].to_numpy() >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "dataset": eligible["dataset"].to_numpy(),
            "y_true": eligible["head_a_label"].astype(int).to_numpy(),
            "y_pred": y_pred,
        }
    )


def build_scoreboard(
    pooled_train: pd.DataFrame,
    model=None,
    heldout_pooled: pd.DataFrame | None = None,
    random_state: int = 0,
) -> dict:
    """Assemble per-dataset Head A metrics. Train-pool via LOSO; held-out via the frozen model."""
    loso = loso_predictions_by_dataset(pooled_train, random_state)

    per_dataset = {}
    for dataset, group in loso.groupby("dataset"):
        per_dataset[str(dataset)] = {
            "evaluation": "grouped-LOSO (in training pool)",
            **_binary_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
        }

    if model is not None and heldout_pooled is not None:
        held = heldout_predictions(model, heldout_pooled)
        for dataset, group in held.groupby("dataset"):
            per_dataset[str(dataset)] = {
                "evaluation": "held-out (frozen model, never trained on)",
                **_binary_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }

    return {
        "head": "head_a",
        "task": "binary graded stress (baseline vs. stress)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_dataset": per_dataset,
        "note": (
            "Train-pool datasets (WESAD, Stress-Predict) are scored by grouped leave-one-subject-out, "
            "not by predicting their own training rows. Nurse Stress is genuinely held out. "
            "Naturalistic Nurse numbers are expected to be lower than lab-protocol datasets: labels "
            "are sparse self-report and the setting is uncontrolled real hospital work."
        ),
    }


def render_scoreboard_markdown(scoreboard: dict) -> str:
    lines = [
        "# Three-dataset validation scoreboard (Head A: graded stress)",
        "",
        f"Generated: {scoreboard.get('generated_at', 'n/a')}",
        f"Task: {scoreboard['task']}",
        "",
        "| Dataset | Evaluation | Balanced acc. | Macro F1 | Windows |",
        "|---|---|---|---|---|",
    ]
    for dataset, m in sorted(scoreboard["per_dataset"].items()):
        lines.append(
            f"| {dataset} | {m['evaluation']} | {m['balanced_accuracy']:.3f} | "
            f"{m['macro_f1']:.3f} | {m['n_windows']} |"
        )
    lines += ["", "## Note", "", scoreboard["note"]]
    return "\n".join(lines)


def save_scoreboard(
    scoreboard: dict, json_path: Path = DEFAULT_SCOREBOARD_JSON, md_path: Path = DEFAULT_SCOREBOARD_MD
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(scoreboard, f, indent=2)
    md_path.write_text(render_scoreboard_markdown(scoreboard))


if __name__ == "__main__":
    raise SystemExit(
        "Run via the script so the frozen model unpickles correctly: "
        "uv run python scripts/build_scoreboard.py"
    )
