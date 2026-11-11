"""Direction 1B: Classifier Chains for Modeling Label Dependencies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import FeatureUnion

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model

logger = logging.getLogger(__name__)


@register_model("ml_classifier_chains")
class ClassifierChainsModel(BaseTopicClassifier):
    """Classifier Chains model to capture topic correlations in tweets."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=40000, sublinear_tf=True)
        self.vectorizer = FeatureUnion([("word", word_vec), ("char", char_vec)])
        
        base_clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
        order = config.get("chain_order", "random")
        self.chain = ClassifierChain(base_clf, order=order, random_state=42)

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> ClassifierChainsModel:
        logger.info("Transforming texts into TF-IDF representation...")
        X_train = self.vectorizer.fit_transform(train_texts)
        logger.info("Fitting ClassifierChain on %d samples...", len(train_texts))
        self.chain.fit(X_train, train_labels)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.chain.predict_proba(X)

    def save(self, output_path: Union[str, Path]) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "chain": self.chain}, output_path)

    def load(self, model_path: Union[str, Path]) -> None:
        data = joblib.load(model_path)
        self.vectorizer = data["vectorizer"]
        self.chain = data["chain"]
