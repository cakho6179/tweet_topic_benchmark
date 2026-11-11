"""Add-on Example 2: Twitter Text Cleaner & Hashtag Normalizer Preprocessor."""

from __future__ import annotations

import re
from typing import List
from tweet_topic.core.registry import register_preprocessor


@register_preprocessor("tweet_text_cleaner")
class TweetTextCleaner:
    """Cleans social media noise while preserving semantic hashtags and topics."""

    def __init__(self, config=None):
        self.config = config or {}

    def transform(self, texts: List[str]) -> List[str]:
        cleaned = []
        for text in texts:
            # Normalize Cardiff mention masks '{@user@}' -> '@user'
            t = re.sub(r"\{@(.*?)@\}", r"@\1", text)
            # Normalize Cardiff URL masks '{{URL}}' -> 'http_url'
            t = t.replace("{{URL}}", "http_url")
            # Unpack camelCase hashtags: #ClimateChange -> #Climate Change
            t = re.sub(r"#([a-z]+)([A-Z])", r"#\1 \2", t)
            # Remove repeated exclamation/question marks
            t = re.sub(r"([!?.]){2,}", r"\1", t)
            # Clean whitespace
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return cleaned
