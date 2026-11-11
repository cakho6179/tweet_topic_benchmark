"""Extensible Registry for Models, Losses, Preprocessors, and Add-ons."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from tweet_topic.core.base import BaseTopicClassifier

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: Dict[str, Type[BaseTopicClassifier]] = {}
_LOSS_REGISTRY: Dict[str, Any] = {}
_PREPROCESSOR_REGISTRY: Dict[str, Any] = {}

# Lazy-load mapping for standard built-in models
_MODEL_MODULE_MAP = {
    "ml_linear_ovr": "tweet_topic.models.ml.linear_model",
    "ml_classifier_chains": "tweet_topic.models.ml.chains_model",
    "ml_lightgbm": "tweet_topic.models.ml.gbm_model",
    "bert_encoder": "tweet_topic.models.bert.encoder_model",
    "decision_d1_local": "tweet_topic.models.decision.local_d1",
    "decision_d1_probe": "tweet_topic.models.decision.d1_probe",
    "decision_d1_api": "tweet_topic.models.decision.api_d1",
    "llm_qwen3_5_4b": "tweet_topic.models.llm.qwen_engine",
    "llm_qwen_qlora_cls": "tweet_topic.models.llm.qlora_classifier",
}

_LOSS_MODULE_MAP = {
    "bce": "tweet_topic.models.bert.losses",
    "focal": "tweet_topic.models.bert.losses",
    "asymmetric": "tweet_topic.models.bert.losses",
}


def register_model(name: str):
    """Decorator to register a topic classification model class."""
    def decorator(cls: Type[BaseTopicClassifier]):
        if name in _MODEL_REGISTRY:
            logger.warning("Overwriting existing model registration for '%s'", name)
        _MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls
    return decorator


def register_loss(name: str):
    """Decorator to register a multi-label loss function."""
    def decorator(cls_or_fn: Any):
        _LOSS_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator


def register_preprocessor(name: str):
    """Decorator to register a custom preprocessor."""
    def decorator(cls_or_fn: Any):
        _PREPROCESSOR_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator


def get_model_cls(name: str) -> Type[BaseTopicClassifier]:
    """Retrieves a registered model class by name (supports lazy-loading)."""
    if name not in _MODEL_REGISTRY:
        if name in _MODEL_MODULE_MAP:
            importlib.import_module(_MODEL_MODULE_MAP[name])
        else:
            discover_addons()

        if name not in _MODEL_REGISTRY:
            available = list(_MODEL_REGISTRY.keys()) + list(_MODEL_MODULE_MAP.keys())
            raise KeyError(f"Model '{name}' not found in registry. Available models: {available}")
    return _MODEL_REGISTRY[name]


def get_preprocessor(name: str) -> Any:
    """Retrieves a registered preprocessor class by name (add-ons are discovered on demand)."""
    if name not in _PREPROCESSOR_REGISTRY:
        discover_addons()
    if name not in _PREPROCESSOR_REGISTRY:
        raise KeyError(f"Preprocessor '{name}' not found. Available: {list(_PREPROCESSOR_REGISTRY)}")
    return _PREPROCESSOR_REGISTRY[name]


def get_loss_fn(name: str) -> Any:
    """Retrieves a registered loss function by name."""
    if name not in _LOSS_REGISTRY:
        if name in _LOSS_MODULE_MAP:
            importlib.import_module(_LOSS_MODULE_MAP[name])

        if name not in _LOSS_REGISTRY:
            available = list(_LOSS_REGISTRY.keys()) + list(_LOSS_MODULE_MAP.keys())
            raise KeyError(f"Loss '{name}' not found in registry. Available losses: {available}")
    return _LOSS_REGISTRY[name]


def list_registered_models() -> Dict[str, Type[BaseTopicClassifier]]:
    return dict(_MODEL_REGISTRY)


def discover_addons(addons_dir: Optional[Path] = None) -> None:
    """Auto-discovers and imports all Python modules in addons/ directory."""
    if addons_dir is None:
        addons_dir = Path(__file__).resolve().parent.parent / "addons"

    if not addons_dir.exists():
        return

    for item in addons_dir.glob("*.py"):
        if item.name.startswith("__"):
            continue
        module_name = f"tweet_topic.addons.{item.stem}"
        try:
            importlib.import_module(module_name)
            logger.debug("Successfully loaded add-on: %s", module_name)
        except Exception as e:
            logger.warning("Failed to load add-on module %s: %s", module_name, e)
