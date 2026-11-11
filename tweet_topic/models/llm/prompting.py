"""Prompt engineering templates for Qwen3.5-4B Multi-Label Classification."""

from __future__ import annotations

from typing import Dict, List, Optional
from tweet_topic.data.taxonomy import TOPIC_CLASSES, TOPIC_DESCRIPTIONS

TAXONOMY_TEXT = "\n".join([f"- {t}: {TOPIC_DESCRIPTIONS.get(t, '')}" for t in TOPIC_CLASSES])


def get_zero_shot_prompt(tweet_text: str) -> List[Dict[str, str]]:
    sys = (
        "You are an expert NLP classifier specialized in Twitter topic analysis.\n"
        "Your task is to classify the tweet into ALL matching topics from the predefined list:\n"
        f"{TAXONOMY_TEXT}\n\n"
        "Output Requirements:\n"
        "1. Output ONLY a valid JSON list containing the exact matching topic string(s).\n"
        "2. If no topics apply, output [].\n"
        "3. Example: [\"sports\", \"celebrity_&_pop_culture\"]\n"
        "4. DO NOT provide explanations, markdown code blocks, or greetings."
    )
    user = f"Classify this tweet:\n\"{tweet_text}\""
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def get_cot_prompt(tweet_text: str) -> List[Dict[str, str]]:
    sys = (
        "You are an expert social media analyst.\n"
        "Analyze the tweet in two steps:\n"
        "Step 1: Briefly list the key topical keywords or entities observed.\n"
        "Step 2: Output the final topics as a strict JSON list at the end.\n"
        f"Allowed categories:\n{', '.join(TOPIC_CLASSES)}"
    )
    user = f"Analyze and classify this tweet:\n\"{tweet_text}\""
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def get_few_shot_prompt(tweet_text: str, exemplars: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert tweet classifier. Classify tweets into zero or more of these 19 topics:\n"
                f"{', '.join(TOPIC_CLASSES)}\n"
                "Respond ONLY with a JSON list of matching topic names."
            ),
        }
    ]
    for ex in exemplars:
        messages.append({"role": "user", "content": f"Tweet: \"{ex['text']}\""})
        messages.append({"role": "assistant", "content": str(ex["labels"]).replace("'", '"')})

    messages.append({"role": "user", "content": f"Tweet: \"{tweet_text}\""})
    return messages
