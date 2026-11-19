"""Direction 2: Transformer Encoder Baseline (BERT / RoBERTa / DeBERTa).

Two head variants, selected purely by config:
* ``pooling: cls`` and ``msd_samples: 1``  -> stock ``AutoModelForSequenceClassification`` head.
* anything else                            -> ``_PooledEncoder``: backbone + (attention|mean|cls) pooling
                                              + Multi-Sample-Dropout head.
"""

from __future__ import annotations

import json
import logging
import random
import time
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
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import get_loss_fn, register_model
from tweet_topic.data.taxonomy import NUM_CLASSES
from tweet_topic.evaluation.metrics import compute_metrics
from tweet_topic.evaluation.thresholding import calibrate_per_class_thresholds
from tweet_topic.models._backbone import pool_tokens
from tweet_topic.models.bert.pooling import AttentionPooling, MultiSampleDropoutHead, reduce_multisample_loss

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "bert-base-uncased"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        self.max_length = int(config.get("max_length", 128))
        self.batch_size = int(config.get("batch_size", 32))
        self.eval_batch_size = int(config.get("eval_batch_size", 64))
        self.grad_accum = max(1, int(config.get("grad_accum", 1)))
        self.epochs = int(config.get("epochs", 5))
        self.lr = float(config.get("lr", 2e-5))
        self.head_lr = float(config.get("head_lr", self.lr))
        self.weight_decay = float(config.get("weight_decay", 0.01))
        self.warmup_ratio = float(config.get("warmup_ratio", 0.1))
        self.scheduler_type = config.get("scheduler", "linear")  # linear | cosine
        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.pooling = config.get("pooling", "cls")
        self.msd_samples = int(config.get("msd_samples", 1))
        self.msd_p = float(config.get("msd_p", 0.2))
        self.attn_dim = int(config.get("attn_dim", 256))
        # Ablation control: keep the custom pooling/MSD head even for CLS + one branch, so that
        # pooling comparisons share exactly the same classifier head.
        self.force_custom_head = bool(config.get("force_custom_head", False))
        self.seed = int(config.get("seed", 42))
        self.save_best = bool(config.get("save_best", True))
        self.select_metric = config.get("select_metric", "val_macro_f1_tuned")
        patience = config.get("early_stopping_patience")
        self.early_stopping_patience = int(patience) if patience is not None else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.history: List[Dict[str, Any]] = []
        self.best_epoch: Optional[int] = None

        # Pluggable loss function
        loss_type = config.get("loss_type", "bce")
        loss_params = dict(config.get("loss_params") or {})
        loss_cls = get_loss_fn(loss_type)
        self.criterion = loss_cls(**loss_params)

        logger.info("Initializing Tokenizer & Model '%s' on %s (loss=%s)...", self.model_name, self.device, loss_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.custom_head = self.force_custom_head or not (self.pooling == "cls" and self.msd_samples <= 1)
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

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader, y_true: np.ndarray) -> Dict[str, Any]:
        self.model.eval()
        all_logits = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self._logits(input_ids, attention_mask)
            all_logits.append(logits.float().cpu())
        logits_t = torch.cat(all_logits, dim=0)
        targets_t = torch.as_tensor(y_true, dtype=torch.float32)

        val_loss = float(self.criterion(logits_t.to(self.device), targets_t.to(self.device)).item())
        val_bce = float(nn.functional.binary_cross_entropy_with_logits(logits_t, targets_t).item())
        probs = torch.sigmoid(logits_t).numpy()

        m05 = compute_metrics(y_true, (probs >= 0.5).astype(int))
        taus = calibrate_per_class_thresholds(y_true, probs)
        mtuned = compute_metrics(y_true, (probs >= taus[None, :]).astype(int))
        return {
            "val_loss": round(val_loss, 5),
            "val_bce_loss": round(val_bce, 5),
            "val_micro_f1_05": m05["micro_f1"],
            "val_macro_f1_05": m05["macro_f1"],
            "val_micro_f1_tuned": mtuned["micro_f1"],
            "val_macro_f1_tuned": mtuned["macro_f1"],
        }

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> TransformerEncoderClassifier:
        self.history = []
        self.best_epoch = None
        if self.epochs <= 0:
            logger.info("Epochs = 0, skipping training (zero-shot/evaluation mode).")
            return self

        _set_seed(self.seed)
        if hasattr(self.criterion, "set_pos_weight_from_labels"):
            self.criterion.set_pos_weight_from_labels(train_labels)
            logger.info("Computed train pos_weight (max=%.1f).", self.criterion.pos_weight_max)

        train_ds = _TorchTweetDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        gen = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=gen)

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_ds = _TorchTweetDataset(val_texts, None, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_ds, batch_size=self.eval_batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self._param_groups(), weight_decay=self.weight_decay)
        steps_per_epoch = max(1, (len(train_loader) + self.grad_accum - 1) // self.grad_accum)
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        if self.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        minimise = self.select_metric in {"val_loss", "val_bce_loss"}
        best_score = float("inf") if minimise else float("-inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0

        logger.info("Starting Transformer Encoder training for %d epochs (seed=%d)...", self.epochs, self.seed)
        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            self.model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = self._logits(input_ids, attention_mask)
                    loss = reduce_multisample_loss(self.criterion, logits.float(), labels)

                running_loss += float(loss.item())
                scaler.scale(loss / self.grad_accum).backward()

                if step % self.grad_accum == 0 or step == len(train_loader):
                    if self.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = running_loss / max(1, len(train_loader))
            record: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(avg_loss, 5),
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_head": optimizer.param_groups[-1]["lr"],
                "epoch_time_sec": round(time.perf_counter() - t0, 2),
            }

            if val_loader is not None and val_labels is not None:
                val_stats = self._eval_loader(val_loader, val_labels)
                record.update(val_stats)
                score = float(val_stats[self.select_metric])
                improved = (score < best_score - 1e-5) if minimise else (score > best_score + 1e-5)
                if improved:
                    best_score = score
                    self.best_epoch = epoch
                    bad_epochs = 0
                    if self.save_best:
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad_epochs += 1

                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_05=%.4f | val_macro_tuned=%.4f (%.1fs)",
                    epoch,
                    self.epochs,
                    avg_loss,
                    val_stats["val_loss"],
                    val_stats["val_macro_f1_05"],
                    val_stats["val_macro_f1_tuned"],
                    record["epoch_time_sec"],
                )
            else:
                logger.info("Epoch %d/%d | train_loss=%.4f (%.1fs)", epoch, self.epochs, avg_loss, record["epoch_time_sec"])

            self.history.append(record)
            if self.early_stopping_patience is not None and bad_epochs >= self.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d (best epoch=%s).", epoch, self.best_epoch)
                break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            logger.info("Restored best checkpoint from epoch %s (%s=%.4f).", self.best_epoch, self.select_metric, best_score)

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
                        "force_custom_head": self.force_custom_head,
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
