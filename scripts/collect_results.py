"""Collects finished run artifacts from a Kaggle output pull into ``outputs/<experiment>/``.

``kaggle kernels output`` preserves the notebook's working directory, so a pulled Vòng 3 suite lands at
``outputs/multi_kaggle_results/kernel_v3_loss/tweet_topic_benchmark/outputs/v3_l1_bce/...``. The figure and
post-processing scripts expect the flat ``outputs/v3_l1_bce/`` layout, so this script copies only the JSON
artifacts (never model weights) and skips runs that are already present unless ``--overwrite`` is given.

Usage:
    python scripts/collect_results.py --source outputs/multi_kaggle_results
    python scripts/collect_results.py --source <dir> --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = (
    "metrics_summary.json",
    "training_history.json",
    "test_predictions.json",
    "val_probabilities.json",
    "postprocess.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run artifacts from Kaggle pulls")
    parser.add_argument("--source", default=str(ROOT / "outputs" / "multi_kaggle_results"))
    parser.add_argument("--dest", default=str(ROOT / "outputs"))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run directory")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest_root = Path(args.dest).resolve()
    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")
    dest_root.mkdir(parents=True, exist_ok=True)

    collected = skipped = 0
    for metrics_path in sorted(source.rglob("metrics_summary.json")):
        run_dir = metrics_path.parent
        if dest_root in run_dir.parents or run_dir == dest_root:
            continue
        try:
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] unreadable {metrics_path}: {exc}")
            continue
        name = summary.get("experiment_name", run_dir.name)
        dest = dest_root / name
        if dest.exists() and not args.overwrite:
            print(f"[skip] {name}: already exists at {dest}")
            skipped += 1
            continue
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        for artifact in ARTIFACTS:
            src = run_dir / artifact
            if src.exists():
                shutil.copyfile(src, dest / artifact)
                copied.append(artifact)
        print(f"[ok] {name}: {', '.join(copied)}")
        collected += 1

    print(f"\nCollected {collected} runs, skipped {skipped} existing runs into {dest_root}")


if __name__ == "__main__":
    main()
