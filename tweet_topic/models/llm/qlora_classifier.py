"""Direction 4B: Qwen3.5-4B as a *supervised* multi-label classifier via QLoRA.

Zero-shot prompting asks a 4B generative model to reproduce a 19-way fine-grained taxonomy it was never trained on,
and then to format the answer as JSON. Two sources of error are mixed together: understanding and generation.
This module removes the generation step and teaches the model the taxonomy:

    tweet -> Qwen3.5-4B (4-bit NF4, frozen, + LoRA adapters) -> pooled state -> MSD head -> 19 sigmoid outputs

* QLoRA: the backbone weights are quantised to 4-bit NormalFloat and frozen; only low-rank adapters (rank r) on the
  attention/MLP projections and the small head are trained, so a 4B model fits a 16 GB T4.
* One forward pass per tweet, no autoregressive decoding, so no JSON-format failures and a fixed, small latency.
* The same threshold-tuning and metrics protocol as every other model applies afterwards.

Requires ``peft`` and ``bitsandbytes`` (installed by the Kaggle runner notebook).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model
from tweet_topic.data.taxonomy import NUM_CLASSES
from tweet_topic.models._backbone import extract_hidden, infer_hidden_size, load_backbone, pool_tokens
from tweet_topic.models.bert.pooling import MultiSampleDropoutHead, reduce_multisample_loss

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _QLoRAClassifierNet(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, pooling: str, msd_p: Sequence[float]):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.head = MultiSampleDropoutHead(hidden_size, NUM_CLASSES, num_samples=len(msd_p), dropout=list(msd_p))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled = pool_tokens(extract_hidden(out).float(), attention_mask, self.pooling)
        return self.head(pooled.to(self.head.fc.weight.dtype))


@register_model("llm_qwen_qlora_cls")
class QwenQLoRAClassifier(BaseTopicClassifier):
    """QLoRA multi-label classifier on Qwen3.5-4B (4-bit NF4 + LoRA + Multi-Sample-Dropout head)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "Qwen/Qwen3.5-4B")
        self.max_length = int(config.get("max_length", 128))
        self.batch_size = int(config.get("batch_size", 16))
        self.eval_batch_size = int(config.get("eval_batch_size", 32))
        self.grad_accum = int(config.get("grad_accum", 2))
        self.epochs = int(config.get("epochs", 3))
        self.lr = float(config.get("lr", 2e-4))
        self.weight_decay = float(config.get("weight_decay", 0.01))
        self.pooling = config.get("pooling", "mean")
        self.msd_p = config.get("msd_p", [0.1, 0.2, 0.3, 0.4])
        self.lora_r = int(config.get("lora_r", 16))
        self.lora_alpha = int(config.get("lora_alpha", 32))
        self.lora_dropout = float(config.get("lora_dropout", 0.05))
        self.target_modules = config.get("target_modules", DEFAULT_TARGETS)
        self.pos_weight_max = float(config.get("pos_weight_max", 30.0))
        self.seed = int(config.get("seed", 42))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.net: Optional[_QLoRAClassifierNet] = None

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading %s in 4-bit NF4 with LoRA (r=%d)...", self.model_name, self.lora_r)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        backbone = load_backbone(
            self.model_name,
            dtype=torch.float16,
            device_map={"": 0} if torch.cuda.is_available() else None,
            prefer="base",
            quantization_config=bnb,
        )
        backbone = prepare_model_for_kbit_training(backbone)
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=list(self.target_modules),
                bias="none",
            ),
        )
        sample = self.tokenizer("hello world", return_tensors="pt").to(next(backbone.parameters()).device)
        hidden_size = infer_hidden_size(backbone, {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]})
        self.net = _QLoRAClassifierNet(backbone, hidden_size, self.pooling, self.msd_p)
        self.net.head.to(self.device)
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.net.parameters())
        logger.info("Trainable params: %d / %d (%.3f%%)", trainable, total, 100.0 * trainable / total)

    def _batches(self, texts: List[str], batch_size: int, labels: Optional[np.ndarray] = None, order=None):
        order = np.arange(len(texts)) if order is None else order
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            enc = self.tokenizer(
                [texts[j] for j in idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            item = {
                "input_ids": enc["input_ids"].to(self.device),
                "attention_mask": enc["attention_mask"].to(self.device),
            }
            if labels is not None:
                item["labels"] = torch.as_tensor(labels[idx], dtype=torch.float32, device=self.device)
            yield item

    # ------------------------------------------------------------------ train / predict
    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> QwenQLoRAClassifier:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        if self.net is None:
            self._build()

        labels = np.asarray(train_labels, dtype=np.float32)
        n_pos = labels.sum(axis=0)
        pos_weight = np.minimum((len(labels) - n_pos) / np.maximum(n_pos, 1.0), self.pos_weight_max)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=self.device))

        params = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        steps_per_epoch = int(np.ceil(len(train_texts) / (self.batch_size * self.grad_accum)))
        total_steps = steps_per_epoch * self.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total_steps))) * max(0.0, 1.0 - s / max(1, total_steps))
        )

        use_cuda = self.device.type == "cuda"
        # T4 has no native bf16, so use fp16 autocast with loss scaling (standard QLoRA-on-T4 practice).
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

        def _optimizer_step() -> None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            order = rng.permutation(len(train_texts))
            running, n_batches = 0.0, 0
            optimizer.zero_grad()
            for step, batch in enumerate(self._batches(train_texts, self.batch_size, labels, order), start=1):
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = self.net(batch["input_ids"], batch["attention_mask"])
                loss = reduce_multisample_loss(criterion, logits.float(), batch["labels"])
                scaler.scale(loss / self.grad_accum).backward()
                running += loss.item()
                n_batches += 1
                if step % self.grad_accum == 0:
                    _optimizer_step()
                if step % 50 == 0:
                    logger.info("Epoch %d | step %d | loss %.4f", epoch, step, loss.item())
            if n_batches % self.grad_accum != 0:  # flush a trailing partial accumulation window
                _optimizer_step()
            optimizer.zero_grad()
            logger.info("=== Epoch %d/%d - mean loss %.4f ===", epoch, self.epochs, running / max(1, n_batches))
        return self

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model not built. Call fit() first.")
        self.net.eval()
        out = []
        for batch in self._batches(texts, self.eval_batch_size):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                logits = self.net(batch["input_ids"], batch["attention_mask"])
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.vstack(out).astype(np.float32)

    def save(self, output_path: Union[str, Path]) -> None:
        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)
        self.net.backbone.save_pretrained(p / "lora_adapter")
        torch.save(self.net.head.state_dict(), p / "head.pt")
