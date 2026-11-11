"""Direction 1C: Gradient Boosted Trees (LightGBM Multi-Label Baseline)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model

logger = logging.getLogger(__name__)


@register_model("ml_lightgbm")
class LightGBMTopicModel(BaseTopicClassifier):
    """Multi-label LightGBM GBDT classifier."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_features = config.get("max_features", 10000)
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, sublinear_tf=True)
        
        try:
            import lightgbm as lgb
            base_estimator = lgb.LGBMClassifier(
                n_estimators=config.get("n_estimators", 150),
                learning_rate=config.get("learning_rate", 0.08),
                num_leaves=31,
                random_state=42,
                verbosity=-1,
                n_jobs=config.get("n_jobs", 1),
            )
        except ImportError:
            logger.warning("LightGBM not installed, falling back to RandomForestClassifier.")
            from sklearn.ensemble import RandomForestClassifier
            base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        self.model = MultiOutputClassifier(base_estimator, n_jobs=1)

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> LightGBMTopicModel:
        logger.info("Vectorizing texts for GBDT...")
        X_train = self.vectorizer.fit_transform(train_texts)
        logger.info("Fitting MultiOutput GBDT...")
        self.model.fit(X_train, train_labels)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        probs_list = self.model.predict_proba(X)
        # Each estimator returns shape (n_samples, 2); extract prob of positive class (col 1)
        probs = np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(texts)) for p in probs_list])
        return probs

    def save(self, output_path: Union[str, Path]) -> None:
        joblib.dump({"vec": self.vectorizer, "model": self.model}, output_path)

    def load(self, model_path: Union[str, Path]) -> None:
        data = joblib.load(model_path)
        self.vectorizer = data["vec"]
        self.model = data["model"]
