"""Extracts detailed per-class metrics, confusion pairs, and qualitative error examples for V4 BERT."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tweet_topic.addons.custom_preprocessor_addon import TweetTextCleaner  # noqa: E402
from tweet_topic.data.taxonomy import TOPIC_CLASSES  # noqa: E402

OUT_ROOT = ROOT / "outputs"


def _load_preds(folder: str, split: str = "test_2021", mode: str = "default"):
    run_dir = OUT_ROOT / folder
    summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    payload = json.loads((run_dir / "test_predictions.json").read_text(encoding="utf-8"))[split]
    yt = np.asarray(payload["y_true"], dtype=int)
    probs = np.asarray(payload.get("y_probs", payload.get("y_probs_raw")), dtype=float)
    if mode == "shrinkage_tuned":
        tau = np.asarray(summary["threshold_comparison"]["tau_shrinkage"], dtype=float)
        yp = (probs >= tau[None, :]).astype(int)
    else:
        yp = np.asarray(payload["y_pred"], dtype=int)
    return yt, yp, probs, summary


def _top_confusions(yt: np.ndarray, yp: np.ndarray, top_k: int = 8) -> List[Dict]:
    pairs = []
    for i in range(len(TOPIC_CLASSES)):
        fn_mask = (yt[:, i] == 1) & (yp[:, i] == 0)
        n_fn = int(np.sum(fn_mask))
        if n_fn == 0:
            continue
        for j in range(len(TOPIC_CLASSES)):
            if i == j:
                continue
            c_fp = int(np.sum(fn_mask & (yp[:, j] == 1) & (yt[:, j] == 0)))
            if c_fp >= 3:
                pairs.append({
                    "missed_true": TOPIC_CLASSES[i],
                    "missed_total_fn": n_fn,
                    "wrongly_predicted_fp": TOPIC_CLASSES[j],
                    "count": c_fp,
                })
    return sorted(pairs, key=lambda x: x["count"], reverse=True)[:top_k]


def _labels_of(row: np.ndarray) -> List[str]:
    return [TOPIC_CLASSES[i] for i, v in enumerate(row) if int(v) == 1]


def main() -> None:
    ds = load_dataset("cardiffnlp/tweet_topic_multi", split="test_2021")
    texts = [x["text"] for x in ds]
    cleaner = TweetTextCleaner()
    cleaned_texts = cleaner.transform(texts)

    yt, yp_s0, _, _ = _load_preds("v4_s0_stock_bce")
    _, yp_s1, _, _ = _load_preds("v4_s1_custom_head_bce")
    _, yp_s2, _, _ = _load_preds("v4_s2_cleaner_bce")
    _, yp_s3, _, _ = _load_preds("v4_s3_hybrid_pool_msd_bce")
    _, yp_s4, _, _ = _load_preds("v4_s4_damped_wbce")
    _, yp_s5, probs_s5, s4_summary = _load_preds("v4_s4_damped_wbce", mode="shrinkage_tuned")

    per_class_rows = []
    for idx, cname in enumerate(TOPIC_CLASSES):
        sup = int(np.sum(yt[:, idx]))
        row = {"class": cname, "support": sup}
        for tag, yp in [("S0", yp_s0), ("S1", yp_s1), ("S2", yp_s2), ("S3", yp_s3), ("S4", yp_s4), ("S5", yp_s5)]:
            p, r, f1, _ = precision_recall_fscore_support(yt[:, idx], yp[:, idx], average="binary", zero_division=0)
            row[tag] = {
                "p": round(float(p), 4),
                "r": round(float(r), 4),
                "f1": round(float(f1), 4),
                "tp": int(np.sum((yt[:, idx] == 1) & (yp[:, idx] == 1))),
                "fp": int(np.sum((yt[:, idx] == 0) & (yp[:, idx] == 1))),
                "fn": int(np.sum((yt[:, idx] == 1) & (yp[:, idx] == 0))),
            }
        per_class_rows.append(row)

    # Extract concrete examples of improvements and remaining errors
    rescued_tail = []
    rescued_cleaner = []
    rescued_multilabel = []
    remaining_errors = []

    tail_ids = {TOPIC_CLASSES.index(c) for c in ("fashion_&_style", "food_&_dining", "travel_&_adventure", "youth_&_student_life", "learning_&_educational")}
    for i in range(len(texts)):
        lt = _labels_of(yt[i])
        l0, l1, l2, l3, l5 = _labels_of(yp_s0[i]), _labels_of(yp_s1[i]), _labels_of(yp_s2[i]), _labels_of(yp_s3[i]), _labels_of(yp_s5[i])
        if l0 != lt and l5 == lt and any(yt[i, c] == 1 for c in tail_ids) and len(rescued_tail) < 5:
            rescued_tail.append({"idx": i, "text": texts[i], "true": lt, "s0": l0, "s5": l5})
        if l1 != lt and l2 == lt and texts[i] != cleaned_texts[i] and len(rescued_cleaner) < 5:
            rescued_cleaner.append({"idx": i, "raw": texts[i], "cleaned": cleaned_texts[i], "true": lt, "s1": l1, "s2": l2})
        if np.sum(yt[i]) >= 2 and l2 != lt and l3 == lt and np.sum(yp_s2[i]) < np.sum(yt[i]) and len(rescued_multilabel) < 5:
            rescued_multilabel.append({"idx": i, "text": texts[i], "true": lt, "s2": l2, "s3": l3})
        if l5 != lt and len(remaining_errors) < 12:
            remaining_errors.append({"idx": i, "text": texts[i], "true": lt, "s5": l5})

    report = {
        "per_class_test_2021": per_class_rows,
        "confusions_s1": _top_confusions(yt, yp_s1),
        "confusions_s5": _top_confusions(yt, yp_s5),
        "examples_rescued_tail": rescued_tail,
        "examples_rescued_cleaner": rescued_cleaner,
        "examples_rescued_multilabel": rescued_multilabel,
        "examples_remaining_errors_s5": remaining_errors,
        "s5_shrinkage_thresholds": s4_summary["threshold_comparison"]["tau_shrinkage"],
        "s4_global_threshold": s4_summary["threshold_comparison"]["tau_global"],
    }
    dest = OUT_ROOT / "v4_detailed_error_cases.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote detailed error report to {dest}")


if __name__ == "__main__":
    main()
