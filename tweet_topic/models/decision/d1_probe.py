"""Direction 3B: Liquid AI d1 as a frozen feature extractor with a supervised multi-label head ("linear probe").

Why this exists
---------------
Reading 19 numbers straight out of a language model's output logits (the old zero-token approach) measures nothing
about the model's understanding of topics: those logits are scores over *vocabulary tokens*. The standard way to ask
"how much topic information does this backbone represent?" is a **linear probe**:

    tweet -> frozen backbone -> pooled hidden state h -> Linear(h) -> 19 sigmoid probabilities

Only the small head is trained, so the result isolates the quality of the frozen representation. It is the right
object to compare against TF-IDF (B1) and a fine-tuned encoder (B2) *under the same protocol*: same train split,
same validation-based threshold tuning, same full test splits, same metrics.

Implementation notes
--------------------
* The backbone is frozen, so each tweet is encoded **once** and cached; the head then trains on cached vectors
  (seconds) instead of re-running a 2B-parameter model every epoch.
* Features are standardised (hidden states of decoder LMs have a few very large outlier dimensions).
* Multi-Sample Dropout is used only while training the head (identical branches at inference).
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

from tweet_topic.core.base import BaseTopicClassifier
from tweet_topic.core.registry import register_model
from tweet_topic.data.taxonomy import NUM_CLASSES
from tweet_topic.models._backbone import extract_hidden, load_backbone, pool_tokens
from tweet_topic.models.bert.pooling import MultiSampleDropoutHead, reduce_multisample_loss

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Tweet: {text}\nThe topics of this tweet are:"


@register_model("decision_d1_probe")
class LiquidD1Probe(BaseTopicClassifier):
    """Frozen Liquid d1 backbone + trainable multi-label head on cached features."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "LiquidAI/d1-2B")
        self.batch_size = int(config.get("batch_size", 16))
        self.max_length = int(config.get("max_length", 128))
        self.pooling = config.get("pooling", "mean")  # mean | last
        self.layer = int(config.get("layer", -1))  # which hidden-state index to read
        self.prompt_template = config.get("prompt_template", DEFAULT_PROMPT)

        self.head_epochs = int(config.get("head_epochs", 80))
        self.head_lr = float(config.get("head_lr", 1e-3))
        self.head_batch = int(config.get("head_batch", 128))
        self.weight_decay = float(config.get("weight_decay", 1e-2))
        self.msd_p: Sequence[float] = config.get("msd_p", [0.1, 0.2, 0.3, 0.4])
        self.pos_weight_max = float(config.get("pos_weight_max", 30.0))
        self.seed = int(config.get("seed", 42))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.backbone = None
        self.head: Optional[MultiSampleDropoutHead] = None
        self.feat_mean: Optional[torch.Tensor] = None
        self.feat_std: Optional[torch.Tensor] = None
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ backbone
    def _ensure_backbone(self) -> None:
        if self.backbone is not None:
            return
        from transformers import AutoTokenizer

        logger.info("Loading frozen backbone '%s' on %s...", self.model_name, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:  # noqa: BLE001 - multimodal repos may only ship a processor
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer = getattr(proc, "tokenizer", proc)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.backbone = load_backbone(
            self.model_name, dtype=dtype, device_map={"": 0} if self.device == "cuda" else None, prefer="base"
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> np.ndarray:
        """Returns pooled features [N, H] (float32), encoding each distinct text only once."""
        self._ensure_backbone()
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        # Sort by length so batches contain similar-length prompts (less padding, same result).
        missing.sort(key=len)
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i : i + self.batch_size]
            prompts = [self.prompt_template.format(text=t) for t in chunk]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(next(self.backbone.parameters()).device)
            out = self.backbone(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True
            )
            hidden_states = getattr(out, "hidden_states", None)
            hidden = hidden_states[self.layer] if hidden_states else extract_hidden(out)
            pooled = pool_tokens(hidden.float(), enc["attention_mask"], self.pooling).cpu().numpy()
            for t, vec in zip(chunk, pooled):
                self._cache[t] = vec.astype(np.float32)
        return np.stack([self._cache[t] for t in texts])

    # ------------------------------------------------------------------ head
    def _standardise(self, feats: torch.Tensor) -> torch.Tensor:
        return (feats - self.feat_mean) / self.feat_std

    def fit(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> LiquidD1Probe:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x_train = torch.from_numpy(self._encode(train_texts))
        y_train = torch.as_tensor(train_labels, dtype=torch.float32)
        self.feat_mean = x_train.mean(dim=0, keepdim=True)
        self.feat_std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
        x_train = self._standardise(x_train)

        x_val = y_val = None
        if val_texts is not None and val_labels is not None:
            x_val = self._standardise(torch.from_numpy(self._encode(val_texts)))
            y_val = torch.as_tensor(val_labels, dtype=torch.float32)

        hidden = x_train.size(1)
        self.head = MultiSampleDropoutHead(hidden, NUM_CLASSES, num_samples=len(self.msd_p), dropout=list(self.msd_p))
        self.head.to(self.device)

        n_pos = y_train.sum(dim=0)
        pos_weight = ((len(y_train) - n_pos) / n_pos.clamp(min=1.0)).clamp(max=self.pos_weight_max).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        eval_criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.head_lr, weight_decay=self.weight_decay)

        best_state, best_val, patience, bad = None, float("inf"), 10, 0
        n = len(x_train)
        for epoch in range(1, self.head_epochs + 1):
            self.head.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.head_batch):
                idx = perm[i : i + self.head_batch]
                xb, yb = x_train[idx].to(self.device), y_train[idx].to(self.device)
                optimizer.zero_grad()
                loss = reduce_multisample_loss(criterion, self.head(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)

            if x_val is not None:
                self.head.eval()
                with torch.no_grad():
                    val_loss = eval_criterion(self.head(x_val.to(self.device)), y_val.to(self.device)).item()
                if val_loss < best_val - 1e-4:
                    best_val, bad = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in self.head.state_dict().items()}
                else:
                    bad += 1
                if epoch % 10 == 0 or bad == 0:
                    logger.info("Head epoch %d | train %.4f | val BCE %.4f", epoch, total / n, val_loss)
                if bad >= patience:
                    logger.info("Early stopping at epoch %d (best val BCE %.4f).", epoch, best_val)
                    break
            elif epoch % 10 == 0:
                logger.info("Head epoch %d | train %.4f", epoch, total / n)

        if best_state is not None:
            self.head.load_state_dict(best_state)
        self.head.eval()
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("Probe head is not trained. Call fit() first.")
        feats = self._standardise(torch.from_numpy(self._encode(texts)))
        self.head.eval()
        with torch.no_grad():
            logits = self.head(feats.to(self.device))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    def release(self) -> None:
        """Frees the backbone once all features are cached (useful before loading another large model)."""
        self.backbone = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
