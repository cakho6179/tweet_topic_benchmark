"""Probability calibration for multi-label classifiers: temperature scaling and expected calibration error.

What temperature scaling can and cannot do
------------------------------------------
Dividing a logit by a positive temperature ``T_c`` is a **monotone** map of ``p_c``. Therefore:

* It does **not** change the ranking of tweets within a class (AUC / average precision are unchanged).
* If thresholds are then re-tuned *per class* by an **exhaustive** search on the same validation set, the best
  achievable per-class F1 is unchanged: the optimal cut point simply moves along the same monotone curve. With a
  coarse threshold grid (e.g. step 0.05 on the probability scale) the reachable cut points move too, so the tuned
  F1 may shift slightly in either direction; that is a grid artifact, not an improvement.
* It *does* change the probability values, so it improves ECE / log-loss, and it changes results that use a
  *shared* threshold across classes (a global threshold, selective-classification scores, ensembling weights).

So it is a tool for trustworthy probabilities, not a source of F1. Claims about F1 gains must come from the
global-threshold or abstention experiments, never from per-class threshold results.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-7


def probs_to_logits(probs: np.ndarray, eps: float = _EPS) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _binary_nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    p = np.clip(_sigmoid(logits / temperature), _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    per_class: bool = True,
    bounds: tuple = (0.05, 10.0),
) -> Union[float, np.ndarray]:
    """Fits temperature(s) by minimising binary negative log-likelihood on a validation set.

    Returns a scalar (``per_class=False``) or an array of shape [C]. A class with no positive (or no negative)
    validation example is degenerate for NLL, so it falls back to T = 1.0.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    if not per_class:
        res = minimize_scalar(
            lambda t: _binary_nll(logits.ravel(), y_true.ravel(), t), bounds=bounds, method="bounded"
        )
        return float(res.x)

    temps = np.ones(logits.shape[1], dtype=np.float64)
    for c in range(logits.shape[1]):
        yc = y_true[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        res = minimize_scalar(lambda t, c=c, yc=yc: _binary_nll(logits[:, c], yc, t), bounds=bounds, method="bounded")
        temps[c] = res.x
    return temps


def apply_temperature(logits: np.ndarray, temperature: Union[float, np.ndarray]) -> np.ndarray:
    """Returns calibrated probabilities sigmoid(logits / T)."""
    t = np.asarray(temperature, dtype=np.float64)
    return _sigmoid(np.asarray(logits, dtype=np.float64) / t)


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Multi-label ECE.

    ``micro_ece`` pools every (tweet, class) probability into one reliability diagram. Because ~90% of labels are
    negative this is dominated by low-probability bins, so ``macro_ece`` (mean of per-class ECE) is reported too.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    def _ece(y: np.ndarray, p: np.ndarray) -> float:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
        total = len(p)
        ece = 0.0
        for b in range(n_bins):
            mask = idx == b
            if not mask.any():
                continue
            ece += (mask.sum() / total) * abs(y[mask].mean() - p[mask].mean())
        return float(ece)

    per_class = [_ece(y_true[:, c], probs[:, c]) for c in range(y_true.shape[1])]
    return {
        "micro_ece": _ece(y_true.ravel(), probs.ravel()),
        "macro_ece": float(np.mean(per_class)),
    }


def calibration_report(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    eval_probs: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    per_class: bool = True,
) -> Dict[str, object]:
    """Fits T on validation and reports ECE before/after on validation (and optionally on an evaluation split)."""
    val_logits = probs_to_logits(val_probs)
    temp = fit_temperature(val_logits, y_val, per_class=per_class)
    report: Dict[str, object] = {
        "temperature": np.atleast_1d(temp).round(4).tolist(),
        "val_before": expected_calibration_error(y_val, val_probs),
        "val_after": expected_calibration_error(y_val, apply_temperature(val_logits, temp)),
    }
    if eval_probs is not None and y_eval is not None:
        eval_logits = probs_to_logits(eval_probs)
        report["eval_before"] = expected_calibration_error(y_eval, eval_probs)
        report["eval_after"] = expected_calibration_error(y_eval, apply_temperature(eval_logits, temp))
    return report
