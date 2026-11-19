"""Loss functions for Multi-Label Classification: BCE, Weighted BCE, and Focal Loss.

Note: Asymmetric Loss (ASL) was cleared from active experiments and documented in ``doc/Future.md``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
from tweet_topic.core.registry import register_loss


@register_loss("bce")
class StandardBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


@register_loss("weighted_bce")
class WeightedBCELoss(nn.Module):
    """Class-balanced BCEWithLogitsLoss using per-class ``pos_weight = (N - N_pos) / N_pos``."""

    def __init__(
        self,
        pos_weight: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        pos_weight_max: float = 30.0,
    ):
        super().__init__()
        self.pos_weight_max = float(pos_weight_max)
        if pos_weight is not None:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32).clamp(min=1.0, max=self.pos_weight_max)
            self.register_buffer("pos_weight", pw)
        else:
            self.register_buffer("pos_weight", None)

    def set_pos_weight_from_labels(self, train_labels) -> None:
        y = torch.as_tensor(train_labels, dtype=torch.float32)
        n_pos = y.sum(dim=0)
        n_neg = len(y) - n_pos
        pw = (n_neg / n_pos.clamp(min=1.0)).clamp(min=1.0, max=self.pos_weight_max)
        self.pos_weight = pw

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


@register_loss("focal")
class MultiLabelFocalLoss(nn.Module):
    """Sigmoid Focal Loss for multi-label classification (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None or float(alpha) < 0 else float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()
