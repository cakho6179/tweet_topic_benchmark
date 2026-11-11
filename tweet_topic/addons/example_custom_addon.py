"""Example Custom Add-on: Demonstrates how easy it is to extend the framework.

Any Python script placed in `tweet_topic/addons/` is automatically discovered!
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model
from tweet_topic.data.taxonomy import NUM_CLASSES


@register_model("custom_heuristic_addon")
class CustomHeuristicAddon(BaseTopicClassifier):
    """A lightweight keyword-heuristic baseline add-on."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.keywords = {
            16: ["goal", "match", "game", "team", "player", "win", "vs", "league"],  # sports
            11: ["album", "song", "music", "track", "concert", "singer", "band"],     # music
            9:  ["game", "playstation", "xbox", "nintendo", "pc", "steam", "fps"],    # gaming
        }

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None):
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        probs = np.full((len(texts), NUM_CLASSES), 0.05, dtype=np.float32)
        for i, text in enumerate(texts):
            lower = text.lower()
            for class_idx, kw_list in self.keywords.items():
                if any(kw in lower for kw in kw_list):
                    probs[i, class_idx] = 0.85
        return probs
