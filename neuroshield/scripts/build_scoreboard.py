"""Build the three-dataset validation scoreboard (D3). Run as a script (not `python -m`).

Extracts WESAD + Stress-Predict for the grouped-LOSO train-pool rows, loads the frozen multi-head
model, and extracts only the Nurse sessions that can contain labelled windows (via
``sessions_covering_labeled_events``) for the held-out row.

Usage: uv run python scripts/build_scoreboard.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuroshield.data.bundle import (  # noqa: E402
    nurse_stress_session_to_bundle,
    stress_predict_subject_to_bundle,
    wesad_subject_to_bundle,
)
from neuroshield.data.nurse_stress_loader import (  # noqa: E402
    load_nurse_stress_session,
    load_survey_events,
    sessions_covering_labeled_events,
)
from neuroshield.data.stress_predict_loader import load_stress_predict_subject  # noqa: E402
from neuroshield.data.wesad_loader import load_wesad_subject  # noqa: E402
from neuroshield.features.extract import extract_features  # noqa: E402
from neuroshield.features.harmonize import harmonize_labels, pool_harmonized  # noqa: E402
from neuroshield.features.personalize import add_personalized_features  # noqa: E402
from neuroshield.models.multihead import load_multihead_artifact  # noqa: E402
from neuroshield.models.scoreboard import build_scoreboard, render_scoreboard_markdown, save_scoreboard  # noqa: E402


def main() -> int:
    # WESAD is the only training pool (LOSO). Stress-Predict and Nurse are held out and scored by
    # the frozen model.
    train_frames = []
    for i in range(2, 18):
        if i == 12:
            continue
        try:
            b = wesad_subject_to_bundle(load_wesad_subject(f"S{i}"))
        except FileNotFoundError:
            continue
        train_frames.append(harmonize_labels(add_personalized_features(extract_features(b)), "wesad")[0])
    if not train_frames:
        print("no WESAD training data available")
        return 1
    pooled_train = pool_harmonized(train_frames)

    model = None
    heldout_pooled = None
    try:
        model, _ = load_multihead_artifact()
        heldout_frames = []
        for i in range(1, 36):
            try:
                b = stress_predict_subject_to_bundle(load_stress_predict_subject(f"S{i:02d}"))
            except FileNotFoundError:
                continue
            heldout_frames.append(harmonize_labels(add_personalized_features(extract_features(b)), "stress_predict")[0])
        events = load_survey_events()
        for sid in sessions_covering_labeled_events(events):
            session = load_nurse_stress_session(sid, events=events)
            bundle = nurse_stress_session_to_bundle(session)
            heldout_frames.append(harmonize_labels(add_personalized_features(extract_features(bundle)), "nurse_stress")[0])
        if heldout_frames:
            heldout_pooled = pool_harmonized(heldout_frames)
            print(f"held-out: {len(heldout_pooled)} windows", flush=True)
    except Exception as exc:  # noqa: BLE001 - held-out rows are optional
        print(f"(held-out rows skipped: {exc})")

    scoreboard = build_scoreboard(pooled_train, model=model, heldout_pooled=heldout_pooled)
    save_scoreboard(scoreboard)
    print(render_scoreboard_markdown(scoreboard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
