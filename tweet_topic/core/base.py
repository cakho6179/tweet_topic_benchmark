"""Abstract Base Class for all Topic Classification Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class BaseTopicClassifier(ABC):
    """Unified interface for all multi-label topic classification models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classes: List[str] = []
        self.threshold: float = 0.5
        self.per_class_thresholds: Optional[np.ndarray] = None

    @abstractmethod
    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> BaseTopicClassifier:
        """Trains or adapts the model on training data, optionally validating."""
        pass

    @abstractmethod
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns continuous probability matrix of shape (n_samples, n_classes) in [0, 1]."""
        pass

    def predict(
        self,
        texts: List[str],
        threshold: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """Returns binary predictions matrix of shape (n_samples, n_classes)."""
        probs = self.predict_proba(texts)
        if threshold is None:
            threshold = self.per_class_thresholds if self.per_class_thresholds is not None else self.threshold

        if isinstance(threshold, np.ndarray):
            return (probs >= threshold[None, :]).astype(int)
        return (probs >= threshold).astype(int)

    def set_thresholds(
        self,
        global_threshold: Optional[float] = None,
        per_class_thresholds: Optional[np.ndarray] = None,
    ) -> None:
        if global_threshold is not None:
            self.threshold = float(global_threshold)
        if per_class_thresholds is not None:
            self.per_class_thresholds = np.asarray(per_class_thresholds, dtype=float)

    def save(self, output_path: Union[str, Path]) -> None:
        """Serializes model artifacts."""
        pass

    def load(self, model_path: Union[str, Path]) -> None:
        """Deserializes model artifacts."""
        pass
