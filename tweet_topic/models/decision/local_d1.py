"""Direction 3A: Liquid AI d1 zero-shot "option-letter" scoring (secondary baseline).

What was wrong before
---------------------
The first version took ``logits[0, -1, :19]``. Those are the language-model scores of the *first 19 token ids of the
vocabulary*; they have no relation to the 19 topics, so the "probabilities" were noise (Micro-F1 ~0.18 at tau=0.70).

What this version does
----------------------
The 19 topics are listed as lettered options (A..S). One forward pass reads the next-token scores of exactly those
19 letter tokens, so each score is genuinely "how much does the model want to answer with this option".

Honest limits
-------------
A single next-token distribution is a *weak* multi-label scorer: it is naturally a single-choice quantity (softmax
over options), so a tweet with three true topics spreads probability mass across them. The softmax share ``s_c``
is mapped to ``19*s_c / (1 + 19*s_c)`` (uniform guess -> 0.5) so thresholds can be searched on a 0..1 scale; this is
a monotone map and does not add information. Treat this as a zero-shot reference point. The linear probe in
``d1_probe.py`` is the meaningful measurement of what the d1 representation knows.
"""

from __future__ import annotations

import logging
import string
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model
from tweet_topic.data.taxonomy import NUM_CLASSES, TOPIC_CLASSES, TOPIC_DESCRIPTIONS
from tweet_topic.models._backbone import load_backbone

logger = logging.getLogger(__name__)

OPTION_LETTERS = list(string.ascii_uppercase[:NUM_CLASSES])


def build_option_prompt(text: str) -> str:
    options = "\n".join(
        f"{letter}. {name} - {TOPIC_DESCRIPTIONS.get(name, '')}" for letter, name in zip(OPTION_LETTERS, TOPIC_CLASSES)
    )
    return (
        "Choose the topic that best describes the tweet.\n\n"
        f"Tweet: \"{text}\"\n\n"
        f"Options:\n{options}\n\n"
        "Answer with a single letter."
    )


@register_model("decision_d1_local")
class LocalLiquidD1Classifier(BaseTopicClassifier):
    """Zero-shot option-letter scoring with a local Liquid AI d1 checkpoint."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "LiquidAI/d1-2B")
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 1024))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._letter_ids: Optional[List[List[int]]] = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoTokenizer

        logger.info("Loading Liquid AI d1 '%s' on %s...", self.model_name, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer = getattr(proc, "tokenizer", proc)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
        self.tokenizer.padding_side = "left"  # so position -1 is the real last token for every row

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = load_backbone(self.model_name, dtype=dtype, device_map={"": 0} if self.device == "cuda" else None)
        self.model.eval()

        # Each letter may tokenise with or without a leading space; both variants are scored and pooled.
        self._letter_ids = []
        for letter in OPTION_LETTERS:
            ids = {
                self.tokenizer.encode(letter, add_special_tokens=False)[0],
                self.tokenizer.encode(" " + letter, add_special_tokens=False)[-1],
            }
            self._letter_ids.append(sorted(ids))

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> LocalLiquidD1Classifier:
        logger.info("Liquid d1 option-letter scoring is zero-shot; nothing to train.")
        return self

    def _render(self, text: str) -> str:
        user = build_option_prompt(text)
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 - tokenizer without a chat template
            return user + "\nAnswer:"

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        n = len(texts)
        probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)

        logger.info("Scoring %d tweets via option letters...", n)
        for i in range(0, n, self.batch_size):
            batch = [self._render(t) for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(next(self.model.parameters()).device)
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            last = out.logits[:, -1, :].float()  # [B, V]
            letter_scores = torch.stack(
                [torch.logsumexp(last[:, ids], dim=-1) for ids in self._letter_ids], dim=-1
            )  # [B, 19]
            share = torch.softmax(letter_scores, dim=-1)
            scaled = NUM_CLASSES * share
            probs[i : i + len(batch)] = (scaled / (1.0 + scaled)).cpu().numpy()
        return probs


@register_model("decision_d1_system_one")
class SystemOneLiquidD1Classifier(BaseTopicClassifier):
    """Native multi-label scoring using LiquidAI/d1-2B ``system_one`` / ``system_one_batch`` API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "LiquidAI/d1-2B")
        self.batch_size = int(config.get("batch_size", 16))
        self.question_mode = config.get("question_mode", "choice_multiple")  # choice_multiple | noul
        self.set_size = config.get("set_size")
        self.dtype_name = str(config.get("dtype", "auto"))  # auto | bfloat16 | float16 | float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._questions = self._build_questions()

    def _pick_dtype(self) -> torch.dtype:
        """bfloat16 only where the hardware has it (compute capability >= 8); T4/P100 fall back to float16."""
        explicit = {
            "float32": torch.float32, "fp32": torch.float32,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
        }
        name = self.dtype_name.lower()
        if name in explicit:
            return explicit[name]
        if self.device == "cpu":
            return torch.float32
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16

    def _build_questions(self) -> Dict[str, Any]:
        if self.question_mode == "noul":
            return {
                cls_name: {
                    "type": "noul",
                    "instructions": (
                        f"Does this tweet discuss the topic '{cls_name.replace('_', ' ')}' "
                        f"({TOPIC_DESCRIPTIONS.get(cls_name, '')})?"
                    ),
                }
                for cls_name in TOPIC_CLASSES
            }
        return {
            "topics": {
                "type": "choice",
                "multiple": True,
                "instructions": "Which topics does this tweet discuss?",
                "criteria": {
                    cls_name: TOPIC_DESCRIPTIONS.get(cls_name, cls_name)
                    for cls_name in TOPIC_CLASSES
                },
            }
        }

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModel

        dtype = self._pick_dtype()
        logger.info(
            "Loading Liquid AI d1 '%s' (native system_one, %s) on %s with dtype %s...",
            self.model_name, self.question_mode, self.device, dtype,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        if not hasattr(self.model, "system_one"):
            raise RuntimeError(
                f"Loaded model '{self.model_name}' does not expose system_one(). Ensure transformers>=5.19 is installed."
            )

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> SystemOneLiquidD1Classifier:
        logger.info("Liquid d1 native system_one is zero-shot; nothing to train.")
        return self

    def _extract_row(self, out: Dict[str, Any]) -> np.ndarray:
        answers = out["answers"]
        if self.question_mode == "noul":
            return np.array([float(answers[c]["noul"]) for c in TOPIC_CLASSES], dtype=np.float32)
        probs_map = answers["topics"]["probabilities"]
        return np.array([float(probs_map.get(c, 0.0)) for c in TOPIC_CLASSES], dtype=np.float32)

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        n = len(texts)
        probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        kwargs: Dict[str, Any] = {}
        if self.set_size is not None:
            kwargs["set_size"] = int(self.set_size)

        logger.info("Running Liquid d1 system_one (%s) on %d tweets...", self.question_mode, n)
        logged_sample = False
        for i in range(0, n, self.batch_size):
            chunk = texts[i : i + self.batch_size]
            if hasattr(self.model, "system_one_batch"):
                outs = self.model.system_one_batch([(t, self._questions) for t in chunk], **kwargs)
            else:
                outs = [self.model.system_one(t, self._questions, **kwargs) for t in chunk]
            if not logged_sample and outs:
                # Full structure of one answer: the only way to confirm how 'probabilities' is nested.
                logger.info("Sample system_one answers (first tweet): %s", repr(outs[0]["answers"])[:2000])
                logged_sample = True
            for j, out in enumerate(outs):
                probs[i + j] = self._extract_row(out)
        if not np.isfinite(probs).all():
            raise RuntimeError("system_one produced non-finite probabilities (check dtype / overflow).")
        if n > 0 and float(probs.max()) <= 0.0:
            raise RuntimeError("system_one probabilities are all zero: the output structure was not parsed.")
        return probs
