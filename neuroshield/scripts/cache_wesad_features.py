"""Extract WESAD features once at a dense window step and cache them to parquet.

Feature extraction (cvxEDA in particular) dominates the runtime of any experiment, so the
accuracy-tuning loop should never pay for it twice. Extracting at a 10s step lets an experiment
subsample to any coarser step that is a multiple of it (30s = every 3rd row), so one cache serves
every window-density variant.

Usage: uv run python scripts/cache_wesad_features.py [--step-sec 10]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from neuroshield.data.bundle import wesad_subject_to_bundle  # noqa: E402
from neuroshield.data.wesad_loader import load_wesad_subject  # noqa: E402
from neuroshield.features.extract import extract_features  # noqa: E402

DEFAULT_CACHE_PATH = Path("artifacts/cache/wesad_features_step10.parquet")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--step-sec", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args()

    frames = []
    for i in range(2, 18):
        if i == 12:
            continue
        try:
            bundle = wesad_subject_to_bundle(load_wesad_subject(f"S{i}"))
        except FileNotFoundError:
            continue
        df = extract_features(bundle, window_sec=args.window_sec, step_sec=args.step_sec)
        frames.append(df)
        print(f"  S{i}: {len(df)} windows", flush=True)

    if not frames:
        print("no WESAD subjects found -- download WESAD first")
        return 1

    pooled = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_parquet(args.out)
    print(f"cached {len(pooled)} windows from {pooled['subject_id'].nunique()} subjects -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
