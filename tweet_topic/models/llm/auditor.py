"""LLM auditor: escalate only the ambiguous tweets to a slower, reasoning model.

Motivation
----------
A fast classifier handles most tweets well; the errors concentrate near decision boundaries. Running a chain-of-thought
LLM on *every* tweet is slow and, as the zero-shot baseline shows, not necessarily accurate. The auditor spends the
expensive model only where uncertainty is highest, and the policy is evaluated honestly: quality of the audited
subset **before vs after**, and overall quality at a given escalation budget.

Policies
--------
replace : the LLM's labels replace the classifier's labels on escalated tweets.
union   : keep classifier labels and add any the LLM proposes (raises recall, can hurt precision).

Whether the auditor *helps* is an empirical question that depends on the LLM; nothing here assumes it does.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union
import numpy as np
from sklearn.metrics import f1_score

from tweet_topic.evaluation.selective import split_by_coverage, uncertainty_scores

logger = logging.getLogger(__name__)


def audit_ambiguous(
    texts: List[str],
    probs: np.ndarray,
    thresholds: Union[float, np.ndarray],
    llm: Any,
    coverage: float = 0.8,
    mode: str = "margin",
    policy: str = "replace",
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Escalates the (1 - coverage) most uncertain tweets to ``llm`` with a chain-of-thought prompt.

    ``llm`` must behave like ``QwenLLMClassifier``: expose ``predict_binary(texts)`` plus the ``prompt_strategy`` and
    ``max_new_tokens`` attributes. Returns the audited predictions, the escalation mask and the classifier's own
    predictions so callers can compare before/after.
    """
    probs = np.asarray(probs, dtype=np.float64)
    tau = np.asarray(thresholds, dtype=np.float64)
    base_preds = (probs >= (tau[None, :] if tau.ndim else tau)).astype(int)

    scores = uncertainty_scores(probs, thresholds, mode)
    keep = split_by_coverage(scores, coverage)
    escalate_idx = np.where(~keep)[0]
    logger.info("Auditor escalating %d / %d tweets (policy=%s).", len(escalate_idx), len(texts), policy)

    audited = base_preds.copy()
    if len(escalate_idx):
        prev_strategy, prev_tokens = llm.prompt_strategy, llm.max_new_tokens
        llm.prompt_strategy, llm.max_new_tokens = "cot", max_new_tokens
        try:
            llm_preds = llm.predict_binary([texts[i] for i in escalate_idx])
        finally:
            llm.prompt_strategy, llm.max_new_tokens = prev_strategy, prev_tokens

        if policy == "replace":
            audited[escalate_idx] = llm_preds
        elif policy == "union":
            audited[escalate_idx] = np.maximum(base_preds[escalate_idx], llm_preds)
        else:
            raise ValueError(f"Unknown audit policy '{policy}'.")

    return {"preds": audited, "escalated": ~keep, "base_preds": base_preds, "scores": scores}


def summarise_audit(y_true: np.ndarray, result: Dict[str, Any]) -> Dict[str, float]:
    """Before/after quality overall and on the escalated subset."""
    y_true = np.asarray(y_true, dtype=int)
    esc = result["escalated"]

    def _f1(y, p):
        return float(f1_score(y, p, average="micro", zero_division=0)) if len(y) else float("nan")

    return {
        "n_escalated": int(esc.sum()),
        "overall_micro_f1_before": _f1(y_true, result["base_preds"]),
        "overall_micro_f1_after": _f1(y_true, result["preds"]),
        "escalated_micro_f1_before": _f1(y_true[esc], result["base_preds"][esc]),
        "escalated_micro_f1_after": _f1(y_true[esc], result["preds"][esc]),
    }
