"""Direction 4: Qwen3.5-4B Large Language Model Multi-Label Classifier."""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model
from tweet_topic.data.taxonomy import NUM_CLASSES
from tweet_topic.models.llm.parser import parse_llm_json_to_multihot
from tweet_topic.models.llm.prompting import get_cot_prompt, get_few_shot_prompt, get_zero_shot_prompt

logger = logging.getLogger(__name__)


@register_model("llm_qwen3_5_4b")
class QwenLLMClassifier(BaseTopicClassifier):
    """Qwen3.5-4B LLM running 4-bit quantized multi-label prompt inference on Kaggle GPU."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "Qwen/Qwen3.5-4B")
        self.load_in_4bit = config.get("load_in_4bit", True)
        self.batch_size = config.get("batch_size", 4)
        self.max_new_tokens = config.get("max_new_tokens", 48)
        self.prompt_strategy = config.get("prompt_strategy", "zero_shot")  # zero_shot | cot | few_shot
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            logger.info("Loading Qwen3.5-4B tokenizer and model (4-bit=%s)...", self.load_in_4bit)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quant_config = None
            if self.load_in_4bit and torch.cuda.is_available():
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self.model.eval()

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> QwenLLMClassifier:
        logger.info("Qwen LLM initialized in prompt inference mode.")
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """LLM binary outputs converted to sharp probability matrix."""
        binary_preds = self.predict_binary(texts)
        # Smooth slightly to allow threshold evaluation
        return np.where(binary_preds == 1, 0.95, 0.05).astype(np.float32)

    def predict_binary(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        n_samples = len(texts)
        preds = np.zeros((n_samples, NUM_CLASSES), dtype=int)

        logger.info("Running Qwen3.5-4B (%s prompt) on %d tweets...", self.prompt_strategy, n_samples)
        for i in range(0, n_samples, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            prompts = []
            for t in batch_texts:
                if self.prompt_strategy == "cot":
                    msgs = get_cot_prompt(t)
                else:
                    msgs = get_zero_shot_prompt(t)
                prompts.append(self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for b_idx, out_seq in enumerate(outputs):
                in_len = inputs["input_ids"][b_idx].shape[0]
                gen_text = self.tokenizer.decode(out_seq[in_len:], skip_special_tokens=True)
                preds[i + b_idx] = parse_llm_json_to_multihot(gen_text)

        return preds
