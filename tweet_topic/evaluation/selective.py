"""Selective classification: decide which tweets to automate and which to escalate.

Idea
----
A multi-label classifier is rarely equally sure about every tweet. If a cheap uncertainty score ranks tweets from
"clearly decided" to "near a decision boundary", we can automate the confident part and escalate the rest to a
human or to a stronger (slower) model. The honest way to evaluate this is a **coverage-quality curve**: keep the
``k%`` most confident tweets, measure quality on that kept part, and see how it changes with ``k``.

Uncertainty scores (higher = less certain)
------------------------------------------
entropy : total binary entropy  H(x) = sum_c H_b(p_c).  Cheap, but a tweet with many *confident negatives* and one
          doubtful label can look "certain" overall, and a model with uniformly low probabilities scores low
          entropy even when it is wrong, so entropy should be validated, not assumed.
margin  : 1 - min_c |p_c - tau_c| / max(tau_c, 1 - tau_c).  A tweet is ambiguous when *any* label sits close to
          its own decision threshold, which is exactly where a prediction can flip.

No number produced here is meaningful until the curve is actually computed on held-out data.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union

import numpy as np
from sklearn.metrics import f1_score

_EPS = 1e-9


def binary_entropy(probs: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Element-wise binary entropy H_b(p) in bits (base 2): 0 at p in {0, 1}, 1 at p = 0.5."""
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0 - _EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(base)


def uncertainty_scores(
    probs: np.ndarray,
    thresholds: Union[float, np.ndarray] = 0.5,
    mode: str = "margin",
) -> np.ndarray:
    """Per-tweet uncertainty, shape [N]; larger means less certain."""
    probs = np.asarray(probs, dtype=np.float64)
    if mode == "entropy":
        return binary_entropy(probs).sum(axis=1)
    if mode == "margin":
        tau = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), probs.shape[1:])
        span = np.maximum(tau, 1.0 - tau)
        closeness = 1.0 - np.abs(probs - tau[None, :]) / span[None, :]
        return closeness.max(axis=1)
    raise ValueError(f"Unknown uncertainty mode '{mode}'. Use 'margin' or 'entropy'.")


def split_by_coverage(scores: np.ndarray, coverage: float) -> np.ndarray:
    """Boolean mask of tweets to automate: the ``coverage`` fraction with the *lowest* uncertainty."""
    n_keep = int(round(np.clip(coverage, 0.0, 1.0) * len(scores)))
    keep = np.zeros(len(scores), dtype=bool)
    if n_keep > 0:
        keep[np.argsort(scores, kind="stable")[:n_keep]] = True
    return keep


def coverage_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: Union[float, np.ndarray] = 0.5,
    mode: str = "margin",
    coverages: Optional[Iterable[float]] = None,
) -> List[Dict[str, float]]:
    """Quality on the automated part as a function of coverage.

    For each coverage reports micro-F1, exact-match and the quality of the *escalated* remainder, so a reader
    can check that the scores really separate easy from hard tweets (escalated part should be clearly worse).
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    tau = np.asarray(thresholds, dtype=np.float64)
    preds = (probs >= (tau[None, :] if tau.ndim else tau)).astype(int)
    scores = uncertainty_scores(probs, thresholds, mode)
    coverages = list(coverages) if coverages is not None else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    def _quality(mask: np.ndarray) -> Dict[str, float]:
        if mask.sum() == 0:
            return {"micro_f1": float("nan"), "exact_match": float("nan")}
        yt, yp = y_true[mask], preds[mask]
        return {
            "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
            "exact_match": float((yt == yp).all(axis=1).mean()),
        }

    rows = []
    for cov in coverages:
        keep = split_by_coverage(scores, cov)
        auto = _quality(keep)
        escalated = _quality(~keep)
        rows.append(
            {
                "coverage": float(cov),
                "n_automated": int(keep.sum()),
                "auto_micro_f1": auto["micro_f1"],
                "auto_exact_match": auto["exact_match"],
                "escalated_micro_f1": escalated["micro_f1"],
                "escalated_exact_match": escalated["exact_match"],
            }
        )
    return rows
