"""Add-on Example 3: Multi-Model Weighted Ensemble Classifier."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.factory import ModelFactory
from tweet_topic.core.registry import register_model
from tweet_topic.data.taxonomy import NUM_CLASSES

logger = logging.getLogger(__name__)


@register_model("custom_weighted_ensemble")
class WeightedEnsembleClassifier(BaseTopicClassifier):
    """Ensemble Add-on combining predictions from multiple registered sub-models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sub_model_configs = config.get("sub_models", [
            {"model_type": "ml_linear_ovr"},
            {"model_type": "custom_heuristic_addon"},
        ])
        self.weights = config.get("weights", [0.7, 0.3])
        self.models: List[BaseTopicClassifier] = [
            ModelFactory.create(sub_cfg) for sub_cfg in self.sub_model_configs
        ]

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> WeightedEnsembleClassifier:
        for idx, m in enumerate(self.models):
            logger.info("Fitting ensemble member %d (%s)...", idx, m.__class__.__name__)
            m.fit(train_texts, train_labels, val_texts, val_labels)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        ensemble_probs = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
        total_w = sum(self.weights)

        for m, w in zip(self.models, self.weights):
            probs = m.predict_proba(texts)
            ensemble_probs += (w / total_w) * probs

        return ensemble_probs
