"""Factory for instantiating models and pipelines from configuration."""

from __future__ import annotations

import logging
from typing import Any, Dict
from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import get_model_cls

logger = logging.getLogger(__name__)


class ModelFactory:
    """Instantiates models dynamically via registry."""

    @staticmethod
    def create(config: Dict[str, Any]) -> BaseTopicClassifier:
        model_type = config.get("model_type")
        if not model_type:
            raise ValueError("Config missing required 'model_type' parameter.")

        cls = get_model_cls(model_type)
        logger.info("Instantiating model '%s' from class %s", model_type, cls.__name__)
        return cls(config)
