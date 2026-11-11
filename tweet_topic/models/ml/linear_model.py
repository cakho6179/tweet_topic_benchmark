"""Direction 1A: Linear OvR Classifier (TF-IDF + Logistic Regression / LinearSVC)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model

logger = logging.getLogger(__name__)


@register_model("ml_linear_ovr")
class LinearOvRClassifier(BaseTopicClassifier):
    """Dual TF-IDF (word + char) with One-vs-Rest Linear Classifier."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        word_ngram = tuple(config.get("word_ngram_range", [1, 2]))
        char_ngram = tuple(config.get("char_ngram_range", [3, 5]))
        max_word = config.get("max_features_word", 40000)
        max_char = config.get("max_features_char", 60000)
        c_param = float(config.get("c_param", 2.0))
        class_weight = config.get("class_weight", "balanced")
        n_jobs = config.get("n_jobs", 1)

        word_vec = TfidfVectorizer(
            ngram_range=word_ngram,
            max_features=max_word,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            max_features=max_char,
            sublinear_tf=True,
            lowercase=True,
        )
        features = FeatureUnion([("word", word_vec), ("char", char_vec)])
        base_clf = LogisticRegression(
            C=c_param,
            max_iter=1000,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=42,
        )
        self.pipeline = Pipeline([("feats", features), ("clf", OneVsRestClassifier(base_clf, n_jobs=n_jobs))])

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> LinearOvRClassifier:
        logger.info("Fitting LinearOvRClassifier on %d samples...", len(train_texts))
        self.pipeline.fit(train_texts, train_labels)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def save(self, output_path: Union[str, Path]) -> None:
        joblib.dump(self.pipeline, output_path)

    def load(self, model_path: Union[str, Path]) -> None:
        self.pipeline = joblib.load(model_path)
