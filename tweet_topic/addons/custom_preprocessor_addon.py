"""Add-on Example 2: Twitter Text Cleaner & BERT-Compatible Normalizer."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

from tweet_topic.core.registry import register_preprocessor

_CASE_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Pure ASCII Unicode-escape map converting common Twitter symbols/emojis to BERT English words
# so bert-base-uncased WordPiece does not collapse topical cues into [UNK].
EMOJI_TOPIC_MAP: Dict[str, str] = {
    # Sports & fitness
    "\u26bd": " soccer ",
    "\u26be": " baseball ",
    "\u26f3": " golf ",
    "\U0001f3c6": " trophy ",
    "\U0001f3c0": " basketball ",
    "\U0001f3c8": " football ",
    "\U0001f3be": " tennis ",
    "\U0001f3c5": " medal ",
    "\U0001f947": " gold medal ",
    "\U0001f4aa": " fitness health ",
    "\U0001f3cb": " fitness workout ",
    "\U0001f3c3": " running fitness ",
    # Music & entertainment
    "\u266a": " music ",
    "\u266b": " music ",
    "\u266c": " music ",
    "\U0001f3b5": " music ",
    "\U0001f3b6": " music ",
    "\U0001f3a4": " music concert ",
    "\U0001f3b8": " music guitar ",
    "\U0001f3ac": " film video ",
    "\U0001f3a5": " film movie ",
    "\U0001f4fa": " tv video ",
    "\U0001f3ae": " video game ",
    "\U0001f579": " video game ",
    "\u2694": " battle game ",
    # Food, travel, fashion, art
    "\u2615": " coffee food ",
    "\U0001f354": " food dining ",
    "\U0001f355": " pizza food ",
    "\U0001f35c": " food dining ",
    "\U0001f377": " wine dining ",
    "\u2708": " travel airplane ",
    "\u26f5": " travel boat ",
    "\U0001f3d6": " travel beach ",
    "\U0001f30d": " world travel ",
    "\U0001f457": " fashion style ",
    "\U0001f484": " beauty fashion ",
    "\U0001f45f": " fashion sneakers ",
    "\U0001f3a8": " art culture ",
    "\u2728": " style sparkle ",
    # Business, tech, education, news
    "\U0001f4b0": " business finance ",
    "\U0001f4c8": " business market ",
    "\U0001f680": " science technology ",
    "\U0001f4bb": " technology software ",
    "\U0001f4da": " learning education ",
    "\U0001f393": " student education ",
    "\u26a0": " news warning ",
    "\U0001f6a8": " breaking news ",
    # Relationships, family, daily emotion
    "\u2764": " love relationship ",
    "\u2665": " love relationship ",
    "\u2661": " love relationship ",
    "\u2763": " love ",
    "\u270a": " solidarity ",
    "\u270c": " peace victory ",
}

_EMOJI_PATTERN = re.compile("|".join(re.escape(k) for k in sorted(EMOJI_TOPIC_MAP, key=len, reverse=True)))


def _split_hashtag(tag: str) -> str:
    """'#ClimateChange' -> '#Climate Change'; '#NBAFinals' -> '#NBA Finals'; '#covid19' stays as is."""
    return "#" + _CASE_BOUNDARY.sub(" ", tag[1:])


@register_preprocessor("tweet_text_cleaner")
class TweetTextCleaner:
    """Cleans social media masks and normalizes hashtags/emojis for Transformer tokenizers."""

    def __init__(self, config=None):
        self.config = config or {}
        self.verbalize_emojis = bool(self.config.get("verbalize_emojis", True))

    def transform(self, texts: List[str]) -> List[str]:
        cleaned = []
        for text in texts:
            t = unicodedata.normalize("NFC", str(text))
            # 1. Strip invisible variation selectors and zero-width joiners first
            t = t.replace("\ufe0f", "").replace("\u200d", "")
            # 2. Normalize Cardiff anonymous user masks and named entity mentions
            t = t.replace("{{USERNAME}}", "@user")
            t = re.sub(r"\{@(.*?)@\}", r"@ \1", t)
            # 3. Normalize URL mask into natural BERT WordPiece tokens
            t = t.replace("{{URL}}", "http link")
            # 4. Verbalize topical emojis so WordPiece does not map them to [UNK]
            if self.verbalize_emojis:
                t = _EMOJI_PATTERN.sub(lambda m: EMOJI_TOPIC_MAP[m.group(0)], t)
            # 5. Unpack camelCase / PascalCase hashtags at case boundaries
            t = re.sub(r"#\w+", lambda m: _split_hashtag(m.group(0)), t)
            # 6. Normalize smart quotes/dashes and collapse repeated punctuation/spaces
            t = t.replace("\u2019", "'").replace("\u2018", "'")
            t = t.replace("\u201c", '"').replace("\u201d", '"')
            t = t.replace("\u2014", " - ").replace("\u2013", " - ")
            t = re.sub(r"([!?.]){2,}", r"\1", t)
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return cleaned


EMOJI_TOPIC_MAP_V2: Dict[str, str] = {
    **EMOJI_TOPIC_MAP,
    # Neutralize heart overexpression so casual likes do not trigger FP relationships
    "\u2764": " love ",
    "\u2665": " love ",
    "\u2661": " love ",
    "\u2728": " ",
    # Missing BMP symbols discovered on validation_2020 [UNK] audit
    "\u263a": " happy ",
    "\u2639": " sad ",
    "\u2753": " ? ",
    "\u2754": " ? ",
    "\u2744": " winter ",
    "\u2746": " winter ",
    "\u260e": " phone ",
    "\u261d": " ",
    "\u2714": " ",
    "\u2705": " ",
}

_EMOJI_KEYS_V2 = sorted(EMOJI_TOPIC_MAP_V2, key=len, reverse=True)
_EMOJI_DEDUP_V2 = re.compile("(" + "|".join(re.escape(k) for k in _EMOJI_KEYS_V2) + r")(?:\s*\1)+")
_EMOJI_PATTERN_V2 = re.compile("|".join(re.escape(k) for k in _EMOJI_KEYS_V2))
_RESIDUAL_SYMBOLS_RE = re.compile(r"[\u2190-\u21FF\u25A0-\u27BF\u2900-\u297F\u2B00-\u2BFF]+")


@register_preprocessor("tweet_text_cleaner_v2")
class TweetTextCleanerV2:
    """Validation-driven v2 normalizer eliminating WordPiece [UNK] tokens & heart bias."""

    def __init__(self, config=None):
        self.config = config or {}
        self.verbalize_emojis = bool(self.config.get("verbalize_emojis", True))

    def transform(self, texts: List[str]) -> List[str]:
        cleaned = []
        for text in texts:
            t = unicodedata.normalize("NFKC", str(text))
            for ch in ("\ufe0f", "\u200d", "\u200b", "\u200c", "\u20e3", "\ufeff", "\ufffd"):
                t = t.replace(ch, "")
            t = t.replace("\u13a9", "A")
            t = t.replace("{{USERNAME}}", "@user")
            t = re.sub(r"\{@(.*?)@\}", r"@ \1", t)
            t = t.replace("{{URL}}", "http link")
            for src, dst in (
                ("\u2019", "'"), ("\u2018", "'"),
                ("\u201c", '"'), ("\u201d", '"'),
                ("\u275d", '"'), ("\u275e", '"'),
                ("\u00ab", '"'), ("\u00bb", '"'),
                ("\u2014", " - "), ("\u2013", " - "),
                ("\u2026", "..."), ("\u2022", " "),
            ):
                t = t.replace(src, dst)
            if self.verbalize_emojis:
                t = _EMOJI_DEDUP_V2.sub(r"\1", t)
                t = _EMOJI_PATTERN_V2.sub(lambda m: EMOJI_TOPIC_MAP_V2[m.group(0)], t)
            t = _RESIDUAL_SYMBOLS_RE.sub(" ", t)
            t = "".join(" " if unicodedata.category(ch) == "So" else ch for ch in t)
            t = re.sub(r"#\w+", lambda m: _split_hashtag(m.group(0)), t)
            t = re.sub(r"([!?.]){2,}", r"\1", t)
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return cleaned
