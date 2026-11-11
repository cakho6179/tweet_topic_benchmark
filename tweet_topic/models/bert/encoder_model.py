"""Direction 2: Transformer Encoder Baseline (RoBERTa / DeBERTa / Twitter-RoBERTa).

Two head variants, selected purely by config (defaults reproduce the original baseline exactly):

* ``pooling: cls`` and ``msd_samples: 1``  -> stock ``AutoModelForSequenceClassification`` head.
* anything else                            -> ``_PooledEncoder``: backbone + (attention|mean|cls) pooling
                                              + Multi-Sample-Dropout head.

Note on checkpoints: ``cardiffnlp/twitter-roberta-base-2021-124m-topic-multi`` is *already fine-tuned* on
``tweet_topic_multi`` with ``train_all`` (it contains 2021 data). For a clean in-time / out-of-time protocol use a
pretrain-only checkpoint such as ``cardiffnlp/twitter-roberta-base-dec2020``.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import get_loss_fn, register_model
from tweet_topic.data.taxonomy import NUM_CLASSES
from tweet_topic.models._backbone import pool_tokens
from tweet_topic.models.bert.pooling import AttentionPooling, MultiSampleDropoutHead, reduce_multisample_loss

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "cardiffnlp/twitter-roberta-base-dec2020"


class _TorchTweetDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[np.ndarray], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class _PooledEncoder(nn.Module):
    """Encoder backbone + configurable pooling + Multi-Sample-Dropout head."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_classes: int,
        pooling: str = "attention",
        attn_dim: int = 256,
        msd_samples: int = 5,
        msd_p: float = 0.2,
    ):
        super().__init__()
        if pooling not in {"cls", "mean", "attention"}:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose cls | mean | attention.")
        self.backbone = backbone
        self.pooling = pooling
        self.pool = AttentionPooling(hidden_size, attn_dim) if pooling == "attention" else None
        self.head = MultiSampleDropoutHead(hidden_size, num_classes, num_samples=msd_samples, dropout=msd_p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "attention":
            pooled, _ = self.pool(hidden, attention_mask)
        elif self.pooling == "mean":
            pooled = pool_tokens(hidden, attention_mask, "mean")
        else:
            pooled = hidden[:, 0, :]
        return self.head(pooled)


@register_model("bert_encoder")
class TransformerEncoderClassifier(BaseTopicClassifier):
    """Transformer Encoder for Multi-label Sequence Classification."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", DEFAULT_CHECKPOINT)
        self.max_length = config.get("max_length", 128)
        self.batch_size = config.get("batch_size", 32)
        self.eval_batch_size = config.get("eval_batch_size", 64)
        self.epochs = config.get("epochs", 3)
        self.lr = float(config.get("lr", 2e-5))
        self.head_lr = float(config.get("head_lr", self.lr))
        self.weight_decay = float(config.get("weight_decay", 0.01))
        self.pooling = config.get("pooling", "cls")
        self.msd_samples = int(config.get("msd_samples", 1))
        self.msd_p = float(config.get("msd_p", 0.2))
        self.attn_dim = int(config.get("attn_dim", 256))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pluggable loss function
        loss_type = config.get("loss_type", "bce")
        loss_cls = get_loss_fn(loss_type)
        self.criterion = loss_cls()

        logger.info("Initializing Tokenizer & Model '%s' on %s...", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.custom_head = not (self.pooling == "cls" and self.msd_samples <= 1)
        if self.custom_head:
            backbone = AutoModel.from_pretrained(self.model_name)
            hidden_size = backbone.config.hidden_size
            self.model = _PooledEncoder(
                backbone,
                hidden_size,
                NUM_CLASSES,
                pooling=self.pooling,
                attn_dim=self.attn_dim,
                msd_samples=self.msd_samples,
                msd_p=self.msd_p,
            ).to(self.device)
            logger.info(
                "Custom head: pooling=%s, msd_samples=%d, msd_p=%.2f", self.pooling, self.msd_samples, self.msd_p
            )
        else:
            model_config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=NUM_CLASSES,
                problem_type="multi_label_classification",
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=model_config,
            ).to(self.device)

    def _logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out if torch.is_tensor(out) else out.logits

    def _param_groups(self) -> List[Dict[str, Any]]:
        """Gives freshly-initialised head parameters their own (usually larger) learning rate."""
        if not self.custom_head:
            return [{"params": list(self.model.parameters()), "lr": self.lr}]
        head_params = list(self.model.head.parameters())
        if self.model.pool is not None:
            head_params += list(self.model.pool.parameters())
        head_ids = {id(p) for p in head_params}
        body = [p for p in self.model.parameters() if id(p) not in head_ids]
        return [{"params": body, "lr": self.lr}, {"params": head_params, "lr": self.head_lr}]

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> TransformerEncoderClassifier:
        if self.epochs <= 0:
            logger.info("Epochs = 0, skipping training (zero-shot/evaluation mode).")
            return self

        train_ds = _TorchTweetDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._param_groups(), weight_decay=self.weight_decay)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        logger.info("Starting Transformer Encoder training for %d epochs...", self.epochs)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = self._logits(input_ids, attention_mask)
                    loss = reduce_multisample_loss(self.criterion, logits.float(), labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(train_loader))
            logger.info("Epoch %d/%d - Loss: %.4f", epoch, self.epochs, avg_loss)

        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        ds = _TorchTweetDataset(texts, None, self.tokenizer, self.max_length)
        loader = DataLoader(ds, batch_size=self.eval_batch_size, shuffle=False)
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = self._logits(input_ids, attention_mask)
                    probs = torch.sigmoid(logits.float()).cpu().numpy()
                    all_probs.append(probs)

        return np.vstack(all_probs)

    def save(self, output_path: Union[str, Path]) -> None:
        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)
        if self.custom_head:
            torch.save(self.model.state_dict(), p / "pooled_encoder.pt")
            (p / "head_config.json").write_text(
                json.dumps(
                    {
                        "model_name": self.model_name,
                        "pooling": self.pooling,
                        "msd_samples": self.msd_samples,
                        "msd_p": self.msd_p,
                        "attn_dim": self.attn_dim,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)

    def load(self, model_path: Union[str, Path]) -> None:
        p = Path(model_path)
        if self.custom_head:
            self.model.load_state_dict(torch.load(p / "pooled_encoder.pt", map_location=self.device))
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(p).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(p)
