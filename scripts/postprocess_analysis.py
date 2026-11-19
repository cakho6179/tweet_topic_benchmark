"""Post-hoc analysis of a finished run from its saved probabilities.

Why this exists
---------------
``scripts/run.py`` already writes a ``threshold_comparison`` block, but it is tied to the threshold mode the
config was launched with, and it does not report temperature-scaled metrics or per-split coverage curves.
The Vòng 3 report needs to read *the same* saved probabilities three ways (0.5 / global / per-class) and to
compare raw versus temperature-scaled decisions. Recomputing from ``val_probabilities.json`` and
``test_predictions.json`` keeps every number traceable to one artifact and avoids retraining.

What is computed
----------------
For every run directory:

* global and per-class temperatures fitted on validation (binary NLL);
* raw and calibrated ECE on validation and on each test split;
* 0.5 / global-tuned / per-class-tuned thresholds for raw and calibrated probabilities, with test metrics;
* coverage-quality curves (margin and entropy) on each test split.

Usage
-----
    python scripts/postprocess_analysis.py --all
    python scripts/postprocess_analysis.py --run-dir outputs/v3_p6_cleaner_weighted_attn_msd
    python scripts/postprocess_analysis.py --run-dir outputs/v3_l1_bce --output-dir outputs/v3_results

Each run gets ``postprocess.json`` next to its other artifacts. ``--output-dir`` additionally receives a copy
named ``<experiment>_postprocess.json`` and, when more than one run is processed, a compact
``postprocess_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tweet_topic.evaluation.calibration import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    probs_to_logits,
)
from tweet_topic.evaluation.metrics import compute_metrics  # noqa: E402
from tweet_topic.evaluation.selective import coverage_curve  # noqa: E402
from tweet_topic.evaluation.thresholding import (  # noqa: E402
    calibrate_global_threshold,
    calibrate_per_class_thresholds,
)

DEFAULT_COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
METRIC_KEYS = (
    "micro_f1",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "exact_match_ratio",
    "hamming_loss",
    "per_class_f1",
)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _binarize(probs: np.ndarray, thresholds) -> np.ndarray:
    t = np.asarray(thresholds, dtype=float)
    return (probs >= (t[None, :] if t.ndim else t)).astype(int)


def _compact(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: metrics[k] for k in METRIC_KEYS}


def _thresholds(y_val: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    tau_global, _ = calibrate_global_threshold(y_val, probs, criterion="micro_f1")
    tau_per_class = calibrate_per_class_thresholds(y_val, probs)
    return {
        "global": round(float(tau_global), 4),
        "per_class": np.asarray(tau_per_class, dtype=float).round(4).tolist(),
    }


def _metric_block(y_true: np.ndarray, probs: np.ndarray, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "default_0.5": _compact(compute_metrics(y_true, _binarize(probs, 0.5))),
        "global_tuned": _compact(compute_metrics(y_true, _binarize(probs, thresholds["global"]))),
        "per_class_tuned": _compact(compute_metrics(y_true, _binarize(probs, thresholds["per_class"]))),
    }


def _ece(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return expected_calibration_error(y_true, probs)


def _split_raw_probs(payload: Dict[str, Any]) -> np.ndarray:
    key = "y_probs_raw" if "y_probs_raw" in payload else "y_probs"
    return np.asarray(payload[key], dtype=np.float64)


def analyze_run(run_dir: Path, coverages: Iterable[float]) -> Dict[str, Any]:
    """Recomputes thresholds, calibration and coverage from the artifacts in ``run_dir``."""
    val_path = run_dir / "val_probabilities.json"
    test_path = run_dir / "test_predictions.json"
    if not val_path.exists():
        raise FileNotFoundError(f"{val_path} is missing; the run did not finish evaluation.")
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} is missing; the run did not finish evaluation.")

    val_payload = _load_json(val_path)
    y_val = np.asarray(val_payload["y_true"], dtype=int)
    val_raw = _split_raw_probs(val_payload)

    val_logits = probs_to_logits(val_raw)
    temp_global = float(fit_temperature(val_logits, y_val, per_class=False))
    temp_per_class = np.asarray(fit_temperature(val_logits, y_val, per_class=True), dtype=np.float64)
    val_calibrated = apply_temperature(val_logits, temp_per_class)
    val_calibrated_global = apply_temperature(val_logits, temp_global)

    thresholds = {
        "raw": _thresholds(y_val, val_raw),
        "calibrated": _thresholds(y_val, val_calibrated),
    }
    validation = {
        "raw": _metric_block(y_val, val_raw, thresholds["raw"]),
        "calibrated": _metric_block(y_val, val_calibrated, thresholds["calibrated"]),
    }
    calibration: Dict[str, Any] = {
        "validation": {
            "raw": _ece(y_val, val_raw),
            "global_temperature": _ece(y_val, val_calibrated_global),
            "per_class_temperature": _ece(y_val, val_calibrated),
        }
    }

    test_payloads = _load_json(test_path)
    test: Dict[str, Any] = {}
    coverage: Dict[str, Any] = {}
    for split, payload in test_payloads.items():
        y_test = np.asarray(payload["y_true"], dtype=int)
        raw = _split_raw_probs(payload)
        calibrated = apply_temperature(probs_to_logits(raw), temp_per_class)

        test[split] = {
            "raw": _metric_block(y_test, raw, thresholds["raw"]),
            "calibrated": _metric_block(y_test, calibrated, thresholds["calibrated"]),
        }
        calibration[split] = {
            "raw": _ece(y_test, raw),
            "global_temperature": _ece(y_test, apply_temperature(probs_to_logits(raw), temp_global)),
            "per_class_temperature": _ece(y_test, calibrated),
        }
        coverage[split] = {
            "raw": {
                "margin": coverage_curve(
                    y_test, raw, thresholds["raw"]["per_class"], mode="margin", coverages=coverages
                ),
                "entropy": coverage_curve(y_test, raw, 0.5, mode="entropy", coverages=coverages),
            },
            "calibrated": {
                "margin": coverage_curve(
                    y_test, calibrated, thresholds["calibrated"]["per_class"], mode="margin", coverages=coverages
                ),
                "entropy": coverage_curve(y_test, calibrated, 0.5, mode="entropy", coverages=coverages),
            },
        }

    return {
        "experiment_name": run_dir.name,
        "run_dir": str(run_dir),
        "n_validation": int(len(y_val)),
        "temperature": {
            "global": round(temp_global, 6),
            "per_class": temp_per_class.round(6).tolist(),
        },
        "thresholds": thresholds,
        "validation": validation,
        "test": test,
        "calibration": calibration,
        "coverage": coverage,
    }


def _summary_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view of one run, used by ``postprocess_summary.json`` and the report tables."""
    row: Dict[str, Any] = {
        "experiment_name": payload["experiment_name"],
        "temperature_global": payload["temperature"]["global"],
        "thresholds": payload["thresholds"],
        "validation": {
            "raw_per_class": payload["validation"]["raw"]["per_class_tuned"],
            "calibrated_per_class": payload["validation"]["calibrated"]["per_class_tuned"],
        },
        "test": {},
        "calibration": payload["calibration"],
    }
    for split, blocks in payload["test"].items():
        cov = payload["coverage"].get(split, {}).get("calibrated", {}).get("margin", [])
        at_80 = next((r for r in cov if abs(r["coverage"] - 0.8) < 1e-9), None)
        row["test"][split] = {
            "raw_default": blocks["raw"]["default_0.5"],
            "raw_global": blocks["raw"]["global_tuned"],
            "raw_per_class": blocks["raw"]["per_class_tuned"],
            "calibrated_per_class": blocks["calibrated"]["per_class_tuned"],
            "calibrated_margin_coverage_0.8": at_80,
        }
    return row


def discover_run_dirs() -> List[Path]:
    """All finished ``outputs/v3_*`` run directories, sorted by name."""
    dirs = [
        p
        for p in sorted((ROOT / "outputs").glob("v3_*"))
        if p.is_dir() and (p / "val_probabilities.json").exists() and (p / "test_predictions.json").exists()
    ]
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc analysis from saved probabilities")
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory (repeatable)")
    parser.add_argument("--all", action="store_true", help="Process every finished outputs/v3_* run")
    parser.add_argument("--output-dir", default=None, help="Also copy results to this directory")
    parser.add_argument(
        "--coverages",
        default=None,
        help="Comma-separated coverage levels (default: 1.0,0.9,0.8,0.7,0.6,0.5)",
    )
    args = parser.parse_args()

    coverages = (
        [float(x) for x in args.coverages.split(",")] if args.coverages else list(DEFAULT_COVERAGES)
    )

    if args.run_dir:
        run_dirs = [Path(p).resolve() for p in args.run_dir]
    elif args.all:
        run_dirs = discover_run_dirs()
    else:
        run_dirs = discover_run_dirs()

    if not run_dirs:
        raise SystemExit("No run directories found (expected outputs/v3_*/val_probabilities.json).")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    payloads: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        payload = analyze_run(run_dir, coverages)
        payloads.append(payload)
        (run_dir / "postprocess.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if output_dir is not None:
            (output_dir / f"{run_dir.name}_postprocess.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        print(f"[{run_dir.name}] thresholds(raw per-class) -> test_2021 micro/macro = "
              f"{payload['test'].get('test_2021', {}).get('raw', {}).get('per_class_tuned', {}).get('micro_f1', float('nan')):.4f}"
              f" / {payload['test'].get('test_2021', {}).get('raw', {}).get('per_class_tuned', {}).get('macro_f1', float('nan')):.4f}")

    if output_dir is not None and len(payloads) > 1:
        summary = {p["experiment_name"]: _summary_row(p) for p in payloads}
        (output_dir / "postprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {output_dir / 'postprocess_summary.json'}")


if __name__ == "__main__":
    main()
