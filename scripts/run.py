"""Unified CLI runner for the extensible tweet_topic benchmark.

Pipeline (every stage after ``model`` is optional and driven by the YAML config):

    data -> [preprocessor] -> model.fit -> val probs -> [temperature scaling] -> thresholds (global | per_class)
         -> test probs -> metrics -> [calibration report] -> [coverage curve] -> [LLM audit of ambiguous tweets]

Probabilities are computed **once per split** and thresholded as arrays. (The previous runner called
``predict_proba`` again inside ``model.predict`` for every split, which tripled the cost for LLM/d1 models.)

Run heavy configs on Kaggle (see ``scripts/build_package_notebooks.py``), not on a local machine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

# Add benchmark root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tweet_topic.core.factory import ModelFactory
from tweet_topic.core.registry import get_preprocessor
from tweet_topic.data.dataset import TweetTopicDataModule
from tweet_topic.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    probs_to_logits,
)
from tweet_topic.evaluation.metrics import compute_metrics
from tweet_topic.evaluation.selective import coverage_curve
from tweet_topic.evaluation.thresholding import (
    calibrate_global_threshold,
    calibrate_per_class_thresholds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tweet_topic_runner")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def binarize(probs: np.ndarray, thresholds) -> np.ndarray:
    t = np.asarray(thresholds, dtype=float)
    return (probs >= (t[None, :] if t.ndim else t)).astype(int)


def main():
    parser = argparse.ArgumentParser(description="TweetTopic Benchmark Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test mode")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    exp_name = cfg.get("experiment_name", "experiment")
    out_dir = Path(cfg.get("output_dir", f"./outputs/{exp_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    max_train = 64 if args.smoke else data_cfg.get("max_train_samples")
    max_eval = 32 if args.smoke else data_cfg.get("max_eval_samples")

    logger.info("Initializing Data Module: %s...", data_cfg.get("dataset_name", "cardiffnlp/tweet_topic_multi"))
    data_mod = TweetTopicDataModule(
        dataset_name=data_cfg.get("dataset_name", "cardiffnlp/tweet_topic_multi"),
        train_split=data_cfg.get("train_split", "train_2020"),
        val_split=data_cfg.get("val_split", "validation_2020"),
        test_splits=data_cfg.get("test_splits", ["test_2020", "test_2021"]),
        max_train_samples=max_train,
        max_eval_samples=max_eval,
    ).setup()

    train_texts, y_train = data_mod.get_train_data()
    val_texts, y_val = data_mod.get_val_data()
    test_splits_data = data_mod.get_test_splits()

    # Optional text preprocessing (a registered add-on), applied identically to every split.
    prep_name = cfg.get("preprocessor")
    if prep_name:
        prep = get_preprocessor(prep_name)(cfg.get("preprocessor_config"))
        logger.info("Applying preprocessor '%s'.", prep_name)
        train_texts = prep.transform(train_texts)
        val_texts = prep.transform(val_texts)
        test_splits_data = {k: (prep.transform(t), y, ids) for k, (t, y, ids) in test_splits_data.items()}

    model_cfg = cfg.get("model", {})
    model = ModelFactory.create(model_cfg)

    logger.info("Training model '%s'...", model_cfg.get("model_type"))
    model.fit(train_texts, y_train, val_texts, y_val)

    # ---- probabilities: one pass per split --------------------------------------------------------------
    val_probs_raw = model.predict_proba(val_texts)
    test_probs_raw = {name: model.predict_proba(texts) for name, (texts, _, _) in test_splits_data.items()}

    # ---- optional temperature scaling (fit on validation only) -------------------------------------------
    # Monotone per class => per-class-threshold F1 is unchanged; it improves probability quality (ECE) and
    # helps shared-threshold settings. See tweet_topic/evaluation/calibration.py.
    temp_mode = cfg.get("temperature", "none")  # none | per_class | global
    temperature = None
    val_probs, test_probs = val_probs_raw, dict(test_probs_raw)
    if temp_mode in ("per_class", "global"):
        temperature = fit_temperature(probs_to_logits(val_probs_raw), y_val, per_class=(temp_mode == "per_class"))
        val_probs = apply_temperature(probs_to_logits(val_probs_raw), temperature)
        test_probs = {k: apply_temperature(probs_to_logits(v), temperature) for k, v in test_probs_raw.items()}
        logger.info("Temperature scaling (%s) fitted on validation.", temp_mode)

    # ---- thresholds ------------------------------------------------------------------------------------
    calib_mode = cfg.get("threshold_mode", "global")  # "global" or "per_class"
    if calib_mode == "per_class":
        thresholds = calibrate_per_class_thresholds(y_val, val_probs)
        model.set_thresholds(per_class_thresholds=thresholds)
        logger.info("Per-class threshold calibration applied.")
    else:
        thresholds, _ = calibrate_global_threshold(y_val, val_probs, criterion=cfg.get("criterion", "micro_f1"))
        model.set_thresholds(global_threshold=thresholds)
        logger.info("Optimal global threshold: %.2f", thresholds)

    val_metrics = compute_metrics(y_val, binarize(val_probs, thresholds))
    logger.info("Validation -> Micro-F1: %.4f | Macro-F1: %.4f", val_metrics["micro_f1"], val_metrics["macro_f1"])

    # ---- test evaluation -------------------------------------------------------------------------------
    test_results: Dict[str, Any] = {}
    test_preds_dict: Dict[str, Any] = {}
    for s_name, (test_texts, y_test, test_ids) in test_splits_data.items():
        probs = test_probs[s_name]
        preds = binarize(probs, thresholds)
        metrics = compute_metrics(y_test, preds)
        test_results[s_name] = metrics
        test_preds_dict[s_name] = {
            "ids": test_ids,
            "y_probs": probs.tolist(),
            "y_probs_raw": test_probs_raw[s_name].tolist(),
            "y_pred": preds.tolist(),
            "y_true": y_test.tolist(),
        }
        logger.info(
            "Test [%s] -> Micro-F1: %.4f | Macro-F1: %.4f | Exact Match: %.4f",
            s_name,
            metrics["micro_f1"],
            metrics["macro_f1"],
            metrics["exact_match_ratio"],
        )

    summary: Dict[str, Any] = {
        "experiment_name": exp_name,
        "config": cfg,
        "validation_metrics": val_metrics,
        "test_metrics": test_results,
        "thresholds": np.atleast_1d(thresholds).round(4).tolist(),
    }

    # ---- optional: calibration report ------------------------------------------------------------------
    if cfg.get("calibration_report", temp_mode != "none"):
        calib: Dict[str, Any] = {
            "temperature_mode": temp_mode,
            "temperature": None if temperature is None else np.atleast_1d(temperature).round(4).tolist(),
            "val_raw": expected_calibration_error(y_val, val_probs_raw),
            "val_calibrated": expected_calibration_error(y_val, val_probs),
        }
        for s_name, (_, y_test, _) in test_splits_data.items():
            calib[f"{s_name}_raw"] = expected_calibration_error(y_test, test_probs_raw[s_name])
            calib[f"{s_name}_calibrated"] = expected_calibration_error(y_test, test_probs[s_name])
        summary["calibration"] = calib

    # ---- optional: selective classification (coverage-quality curve) ------------------------------------
    sel_cfg = cfg.get("selective")
    if sel_cfg:
        summary["selective"] = {
            s_name: coverage_curve(
                y_test,
                test_probs[s_name],
                thresholds,
                mode=sel_cfg.get("mode", "margin"),
                coverages=sel_cfg.get("coverages"),
            )
            for s_name, (_, y_test, _) in test_splits_data.items()
        }

    # ---- optional: LLM auditor on the most ambiguous tweets ----------------------------------------------
    audit_cfg = cfg.get("audit")
    if audit_cfg:
        from tweet_topic.models.llm.auditor import audit_ambiguous, summarise_audit

        model.release() if hasattr(model, "release") else None  # free GPU memory for the auditor LLM
        llm = ModelFactory.create(audit_cfg["llm"])
        summary["audit"] = {}
        for s_name, (test_texts, y_test, _) in test_splits_data.items():
            result = audit_ambiguous(
                test_texts,
                test_probs[s_name],
                thresholds,
                llm,
                coverage=audit_cfg.get("coverage", 0.8),
                mode=audit_cfg.get("mode", "margin"),
                policy=audit_cfg.get("policy", "replace"),
            )
            summary["audit"][s_name] = summarise_audit(y_test, result)
            logger.info("Audit [%s] -> %s", s_name, summary["audit"][s_name])

    # ---- persist ---------------------------------------------------------------------------------------
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_preds_dict, f)

    with open(out_dir / "val_probabilities.json", "w", encoding="utf-8") as f:
        json.dump({"y_probs_raw": val_probs_raw.tolist(), "y_true": y_val.tolist()}, f)

    print("\n" + "=" * 65)
    print(f"BENCHMARK COMPLETE: {exp_name}")
    print("=" * 65)
    print(f"Val Micro-F1: {val_metrics['micro_f1']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")
    for s_name, m in test_results.items():
        print(f"[{s_name.upper()}] Micro-F1: {m['micro_f1']:.4f} | Macro-F1: {m['macro_f1']:.4f} | Exact Match: {m['exact_match_ratio']:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
