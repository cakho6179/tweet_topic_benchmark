"""Direction 3B: Liquid AI System One API Decision Model via typesafe-sdk."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model
from tweet_topic.data.taxonomy import NUM_CLASSES, TOPIC_CLASSES, TOPIC_DESCRIPTIONS

logger = logging.getLogger(__name__)


@register_model("decision_d1_api")
class ApiLiquidD1Classifier(BaseTopicClassifier):
    """Queries Liquid AI System One API for calibrated multi-label decision probabilities."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.environ.get("LIQUID_API_KEY", "")
        self.base_url = config.get("base_url", "https://api.liquid.ai/decisions")
        self.model_name = config.get("model_name", "d1.1")
        self.client = None

    def _get_client(self):
        if self.client is None:
            try:
                from typesafe_sdk import TypeSafeClient
                self.client = TypeSafeClient(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model_name,
                )
            except ImportError:
                raise ImportError("Please install typesafe-sdk: pip install typesafe-sdk")
        return self.client

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> ApiLiquidD1Classifier:
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        client = self._get_client()
        from typesafe_sdk import Noul

        questions = {
            cls_name: Noul(instructions=f"Does this tweet discuss {cls_name.replace('_', ' ')}? Context: {TOPIC_DESCRIPTIONS.get(cls_name, '')}")
            for cls_name in TOPIC_CLASSES
        }

        n_samples = len(texts)
        probs = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)

        logger.info("Querying Liquid System One Decision API for %d samples...", n_samples)
        for i, text in enumerate(texts):
            try:
                resp = client.system_one(state=text, questions=questions)
                for c_idx, cls_name in enumerate(TOPIC_CLASSES):
                    ans = resp.answers.get(cls_name)
                    probs[i, c_idx] = float(ans.noul) if ans else 0.05
            except Exception as e:
                logger.error("Decision API error on sample %d: %s", i, e)
                probs[i] = 0.05

        return probs
