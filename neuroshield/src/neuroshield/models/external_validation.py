"""T18: run the frozen WESAD-trained M1 model against an external dataset's windows.

Stress-Predict uses the same wrist channels as WESAD (both are Empatica E4 exports), so every
FEATURE_COLUMNS entry is computable -- unlike a dataset missing a whole channel (e.g. no ACC),
this is a same-channel-set validation. What differs is the labelling: Stress-Predict provides a
per-second binary ground truth (0/1) from a *different* stress protocol (Stroop, interview,
hyperventilation) rather than WESAD's TSST-style stressor, so a good score here is evidence the
model generalizes across stress *inducers*, not just across people.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from neuroshield.data.bundle import stress_predict_subject_to_bundle
from neuroshield.data.stress_predict_loader import load_stress_predict_subject
from neuroshield.features.extract import extract_features
from neuroshield.features.labels import DEFAULT_MIN_VALID_FRACTION
from neuroshield.models.artifact import load_model_artifact, predict_proba_stress

DEFAULT_NOTES_PATH = Path("artifacts/metrics/external_validation_notes.md")

ALL_STRESS_PREDICT_SUBJECTS = [f"S{i:02d}" for i in range(1, 36)]


def _extract_labeled_windows(
    subject_ids: list[str], min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    skipped: dict[str, str] = {}
    for sid in subject_ids:
        try:
            subject = load_stress_predict_subject(sid)
        except FileNotFoundError as exc:
            skipped[sid] = str(exc)
            continue
        bundle = stress_predict_subject_to_bundle(subject)
        features = extract_features(bundle)
        if (features["label"] != -1).sum() == 0:
            skipped[sid] = "no processed ground-truth labels available for this subject"
            continue
        frames.append(features)

    if not frames:
        return pd.DataFrame(), skipped

    all_windows = pd.concat(frames, ignore_index=True)
    kept = all_windows[
        (all_windows["label"].isin([0, 1])) & (all_windows["valid_fraction"] >= min_valid_fraction)
    ].copy()
    kept["m1_label"] = kept["label"].astype(int)
    return kept, skipped


def run_external_validation(
    subject_ids: list[str] = None, min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION
) -> dict:
    subject_ids = subject_ids or ALL_STRESS_PREDICT_SUBJECTS
    pipeline, manifest = load_model_artifact()

    kept, skipped = _extract_labeled_windows(subject_ids, min_valid_fraction)
    if len(kept) == 0:
        return {
            "dataset": "stress_predict",
            "model_version": manifest["model_version"],
            "n_windows_evaluated": 0,
            "skipped_subjects": skipped,
            "compatible": False,
            "incompatibility_reason": "no subject produced any windows with usable ground-truth labels",
        }

    probabilities = predict_proba_stress(pipeline, manifest, kept)
    y_true = kept["m1_label"].to_numpy()
    y_pred = (probabilities >= 0.5).astype(int)

    overall = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    per_subject = {}
    for sid, group in kept.groupby("subject_id"):
        idx = group.index
        pos = kept.index.get_indexer(idx)
        yt, yp = y_true[pos], y_pred[pos]
        per_subject[sid] = {
            "n_windows": int(len(group)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(set(yt)) > 1 else None,
        }

    return {
        "dataset": "stress_predict",
        "model_version": manifest["model_version"],
        "feature_version": manifest["feature_version"],
        "n_subjects_evaluated": int(kept["subject_id"].nunique()),
        "n_windows_evaluated": int(len(kept)),
        "class_balance": {
            "baseline": int((y_true == 0).sum()),
            "stress": int((y_true == 1).sum()),
        },
        "skipped_subjects": skipped,
        "compatible": True,
        "overall": overall,
        "per_subject": per_subject,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def render_notes_markdown(result: dict) -> str:
    lines = [
        "# External validation: Stress-Predict Dataset",
        "",
        f"Generated: {result.get('generated_at', 'n/a')}",
        f"Frozen model: `{result.get('model_version', 'n/a')}` (feature_version `{result.get('feature_version', 'n/a')}`)",
        "",
        "## Channel compatibility",
        "",
        "Stress-Predict was recorded on the same Empatica E4 wrist hardware family as WESAD, so "
        "every channel M1 needs (BVP, EDA, TEMP, ACC) is present and every `features-v1` column is "
        "computable -- this is a same-channel-set external validation, not a partial one. "
        "ACC is in the same raw unit (1/64 g) on both datasets. Ground truth differs: Stress-Predict "
        "labels come from a per-second binary table (0/1) derived from a different stressor protocol "
        "(Stroop colour-word test, interview, hyperventilation) than WESAD's stress task, so this is "
        "a test of generalization across stress *inducers*, not just across people.",
        "",
        "One subject (S01) exists in the raw data but has no entry in the processed label table "
        "(likely an excluded pilot participant) and is skipped for lack of ground truth.",
        "",
    ]

    if not result.get("compatible", False):
        lines += [
            "## Result: incompatible",
            "",
            f"Reason: {result.get('incompatibility_reason', 'unknown')}",
            "",
            f"Skipped subjects: {result.get('skipped_subjects', {})}",
        ]
        return "\n".join(lines)

    overall = result["overall"]
    lines += [
        "## Result: ran through the frozen pipeline",
        "",
        f"- Subjects evaluated: {result['n_subjects_evaluated']}",
        f"- Windows evaluated: {result['n_windows_evaluated']}",
        f"- Class balance: baseline={result['class_balance']['baseline']}, stress={result['class_balance']['stress']}",
        f"- Balanced accuracy: {overall['balanced_accuracy']:.3f}",
        f"- Macro F1: {overall['macro_f1']:.3f}",
        f"- Confusion matrix [[TN,FP],[FN,TP]]: {overall['confusion_matrix']}",
        "",
    ]
    if result.get("skipped_subjects"):
        lines += ["## Skipped subjects", ""]
        for sid, reason in result["skipped_subjects"].items():
            lines.append(f"- `{sid}`: {reason}")
        lines.append("")

    lines += ["## Per-subject balanced accuracy", "", "| Subject | Windows | Balanced accuracy |", "|---|---|---|"]
    for sid, metrics in sorted(result["per_subject"].items()):
        ba = f"{metrics['balanced_accuracy']:.3f}" if metrics["balanced_accuracy"] is not None else "n/a (single class)"
        lines.append(f"| {sid} | {metrics['n_windows']} | {ba} |")

    return "\n".join(lines)


def save_notes(result: dict, path: Path = DEFAULT_NOTES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_notes_markdown(result))


if __name__ == "__main__":
    result = run_external_validation()
    save_notes(result)
    print(f"compatible={result['compatible']}")
    if result["compatible"]:
        print(
            f"n_subjects={result['n_subjects_evaluated']} n_windows={result['n_windows_evaluated']} "
            f"balanced_accuracy={result['overall']['balanced_accuracy']:.3f} "
            f"macro_f1={result['overall']['macro_f1']:.3f}"
        )
    print(f"wrote {DEFAULT_NOTES_PATH}")
