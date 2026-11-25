"""Threshold calibration algorithms for multi-label classification."""

from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score


def calibrate_global_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    search_range: Tuple[float, float, float] = (0.1, 0.9, 0.05),
    criterion: str = "micro_f1",
) -> Tuple[float, float]:
    """Finds a single scalar threshold that maximizes the specified F1 metric."""
    best_t = 0.5
    best_score = -1.0
    start, stop, step = search_range

    for t in np.arange(start, stop + 1e-6, step):
        preds = (y_probs >= t).astype(int)
        score = f1_score(
            y_true,
            preds,
            average="macro" if criterion == "macro_f1" else "micro",
            zero_division=0,
        )
        if score > best_score:
            best_score = float(score)
            best_t = float(t)

    return best_t, best_score


def calibrate_per_class_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    search_range: Tuple[float, float, float] = (0.05, 0.85, 0.05),
    default_threshold: float = 0.5,
) -> np.ndarray:
    """Finds class-specific optimal thresholds maximizing each class's individual F1."""
    n_classes = y_true.shape[1]
    optimal_thresholds = np.full(n_classes, default_threshold, dtype=float)
    start, stop, step = search_range

    for c in range(n_classes):
        best_f1 = -1.0
        best_c_t = default_threshold
        # If class has zero positive labels in val set, retain default
        if np.sum(y_true[:, c]) == 0:
            continue

        for t in np.arange(start, stop + 1e-6, step):
            pred_c = (y_probs[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], pred_c, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_c_t = float(t)

        optimal_thresholds[c] = best_c_t

    return optimal_thresholds


def calibrate_shrinkage_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    shrinkage_k: float = 20.0,
    global_criterion: str = "micro_f1",
    search_range: Tuple[float, float, float] = (0.10, 0.85, 0.02),
) -> np.ndarray:
    """Support-regularized per-class thresholds shrunk toward the global optimum.

    Prevents threshold overfitting on small validation sets by shrinking rare-class thresholds
    toward ``tau_global`` proportional to validation positive support:
        alpha_c = N_val_pos_c / (N_val_pos_c + shrinkage_k)
        tau_c*  = alpha_c * tau_val_c + (1 - alpha_c) * tau_global
    """
    tau_global, _ = calibrate_global_threshold(
        y_true,
        y_probs,
        search_range=search_range,
        criterion=global_criterion,
    )
    n_classes = y_true.shape[1]
    start, stop, step = search_range
    grid = np.arange(start, stop + 1e-6, step)
    shrunk_thresholds = np.full(n_classes, tau_global, dtype=float)

    for c in range(n_classes):
        n_pos = int(np.sum(y_true[:, c]))
        if n_pos == 0:
            shrunk_thresholds[c] = tau_global
            continue

        best_f1 = -1.0
        best_c_t = tau_global
        for t in grid:
            t_val = float(t)
            pred_c = (y_probs[:, c] >= t_val).astype(int)
            f1 = float(f1_score(y_true[:, c], pred_c, zero_division=0))
            if f1 > best_f1 + 1e-9:
                best_f1 = f1
                best_c_t = t_val
            elif abs(f1 - best_f1) <= 1e-9 and abs(t_val - tau_global) < abs(best_c_t - tau_global):
                best_c_t = t_val

        alpha = float(n_pos) / float(n_pos + shrinkage_k)
        shrunk_thresholds[c] = round(alpha * best_c_t + (1.0 - alpha) * tau_global, 4)

    return shrunk_thresholds


def apply_empty_fallback(preds: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Guarantees every tweet receives at least its Top-1 highest-probability topic."""
    out = preds.copy()
    empty_rows = np.where(out.sum(axis=1) == 0)[0]
    for i in empty_rows:
        out[i, int(np.argmax(probs[i]))] = 1
    return out
