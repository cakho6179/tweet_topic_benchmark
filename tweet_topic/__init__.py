"""Tweet Topic Multi-Label Benchmark and Add-on Framework."""

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.factory import ModelFactory
from tweet_topic.core.registry import discover_addons, register_loss, register_model, register_preprocessor

# Auto-discover all add-ons upon import
discover_addons()

__version__ = "1.0.0"
__all__ = [
    "BaseTopicClassifier",
    "ModelFactory",
    "register_model",
    "register_loss",
    "register_preprocessor",
    "discover_addons",
]
