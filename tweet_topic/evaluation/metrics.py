"""Evaluation metrics for multi-label classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from tweet_topic.data.taxonomy import TOPIC_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multi-label evaluation metrics."""
    if classes is None:
        classes = TOPIC_CLASSES

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_p = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    micro_r = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    exact = float(accuracy_score(y_true, y_pred))
    hamming = float(hamming_loss(y_true, y_pred))

    per_class = {
        cls_name: round(float(f1), 4)
        for cls_name, f1 in zip(classes, f1_score(y_true, y_pred, average=None, zero_division=0))
    }

    return {
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "macro_precision": round(macro_p, 4),
        "micro_recall": round(micro_r, 4),
        "macro_recall": round(macro_r, 4),
        "exact_match_ratio": round(exact, 4),
        "hamming_loss": round(hamming, 4),
        "per_class_f1": per_class,
    }
