"""Train and freeze the multi-head model (D2). Run as a script, not `python -m ...`.

Running the training from a standalone script (that imports the model class from its real module)
ensures ``MultiHeadModel`` is pickled as ``neuroshield.models.multihead.MultiHeadModel`` rather
than ``__main__.MultiHeadModel`` -- the latter would be unloadable from any other context.

Usage: uv run python scripts/train_multihead.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuroshield.data.bundle import (  # noqa: E402
    wesad_subject_to_bundle,
)
from neuroshield.data.wesad_loader import load_wesad_subject  # noqa: E402
from neuroshield.features.extract import extract_features  # noqa: E402
from neuroshield.features.harmonize import harmonize_labels, pool_harmonized  # noqa: E402
from neuroshield.features.personalize import add_personalized_features  # noqa: E402
from neuroshield.models.multihead import (  # noqa: E402
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    evaluate_head_a,
    evaluate_head_b,
    save_multihead_artifact,
    train_final_multihead,
)


def main() -> int:
    frames = []
    for i in range(2, 18):
        if i == 12:
            continue
        try:
            b = wesad_subject_to_bundle(load_wesad_subject(f"S{i}"))
        except FileNotFoundError:
            continue
        frames.append(harmonize_labels(add_personalized_features(extract_features(b)), "wesad")[0])
        print(f"  extracted wesad S{i}", flush=True)
    # Head A trains on WESAD only (Stress-Predict/Nurse are held-out validation, not training).

    if not frames:
        print("no WESAD training data available -- download WESAD first")
        return 1

    pooled = pool_harmonized(frames)
    print("evaluating heads (grouped LOSO)...", flush=True)
    metrics = {"head_a": evaluate_head_a(pooled), "head_b": evaluate_head_b(pooled)}
    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    model = train_final_multihead(pooled)
    manifest = save_multihead_artifact(model, pooled)

    print(f"Head A: model bal_acc={metrics['head_a']['model']['balanced_accuracy']:.3f} "
          f"vs dummy {metrics['head_a']['dummy']['balanced_accuracy']:.3f}")
    print(f"Head B: model bal_acc={metrics['head_b']['model']['balanced_accuracy']:.3f} "
          f"vs dummy {metrics['head_b']['dummy']['balanced_accuracy']:.3f}")
    print(f"saved {DEFAULT_MODEL_PATH} (checksum {manifest['checksum_sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
