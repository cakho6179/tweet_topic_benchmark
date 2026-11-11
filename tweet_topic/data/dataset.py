"""Dataset loader and split manager for cardiffnlp/tweet_topic_multi."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datasets import load_dataset

from tweet_topic.data.taxonomy import NUM_CLASSES, TOPIC_CLASSES

logger = logging.getLogger(__name__)


class TweetTopicDataModule:
    """Manages loading, slicing, and formatting of TweetTopic splits."""

    def __init__(
        self,
        dataset_name: str = "cardiffnlp/tweet_topic_multi",
        train_split: str = "train_2020",
        val_split: str = "validation_2020",
        test_splits: Optional[List[str]] = None,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.val_split = val_split
        self.test_splits = test_splits or ["test_2020", "test_2021"]
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.raw_dataset = None

    def setup(self) -> TweetTopicDataModule:
        logger.info("Loading '%s' from Hugging Face Hub...", self.dataset_name)
        self.raw_dataset = load_dataset(self.dataset_name)
        return self

    def get_train_data(self) -> Tuple[List[str], np.ndarray]:
        split = self.raw_dataset[self.train_split]
        if self.max_train_samples and self.max_train_samples < len(split):
            split = split.select(range(self.max_train_samples))
        texts = [x["text"] for x in split]
        labels = np.array([x["label"] for x in split], dtype=np.float32)
        return texts, labels

    def get_val_data(self) -> Tuple[List[str], np.ndarray]:
        split = self.raw_dataset[self.val_split]
        if self.max_eval_samples and self.max_eval_samples < len(split):
            split = split.select(range(self.max_eval_samples))
        texts = [x["text"] for x in split]
        labels = np.array([x["label"] for x in split], dtype=np.float32)
        return texts, labels

    def get_test_splits(self) -> Dict[str, Tuple[List[str], np.ndarray, List[str]]]:
        results = {}
        for s in self.test_splits:
            split = self.raw_dataset[s]
            if self.max_eval_samples and self.max_eval_samples < len(split):
                split = split.select(range(self.max_eval_samples))
            texts = [x["text"] for x in split]
            labels = np.array([x["label"] for x in split], dtype=np.float32)
            ids = [x.get("id", str(i)) for i, x in enumerate(split)]
            results[s] = (texts, labels, ids)
        return results
