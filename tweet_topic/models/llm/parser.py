"""Parser for LLM generated output to binary multi-hot vectors."""

from __future__ import annotations

import json
import re
from typing import List
import numpy as np
from tweet_topic.data.taxonomy import NUM_CLASSES, TOPIC_CLASSES


def parse_llm_json_to_multihot(raw_output: str) -> np.ndarray:
    """Extracts topic labels from LLM output into a 19-dimensional binary vector."""
    vector = np.zeros(NUM_CLASSES, dtype=int)
    cleaned = raw_output.strip()

    # Try JSON arrays, last one first: a chain-of-thought answer puts the final list at the end, and the
    # reasoning before it may itself contain brackets. For single-array outputs this is identical to before.
    for candidate in reversed(re.findall(r"\[.*?\]", cleaned, re.DOTALL)):
        try:
            items = json.loads(candidate)
        except Exception:
            continue
        if isinstance(items, list):
            for item in items:
                norm = str(item).strip().lower().replace(" ", "_")
                for idx, c in enumerate(TOPIC_CLASSES):
                    if norm == c.lower():
                        vector[idx] = 1
            return vector

    # Fallback to robust token search
    lower_text = cleaned.lower()
    for idx, c in enumerate(TOPIC_CLASSES):
        token_pattern = r"\b" + re.escape(c.lower()) + r"\b"
        human_name = c.replace("_&_", " & ").replace("_", " ").lower()
        human_pattern = r"\b" + re.escape(human_name) + r"\b"
        if re.search(token_pattern, lower_text) or re.search(human_pattern, lower_text):
            vector[idx] = 1

    return vector
