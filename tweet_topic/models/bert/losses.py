"""Loss functions for Multi-Label Classification: BCE, Damped Weighted BCE, Focal Loss, and Asymmetric Loss."""

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
    """Class-balanced BCEWithLogitsLoss with configurable power-damping: ``((N - N_pos) / N_pos)^power``."""

    def __init__(
        self,
        pos_weight: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        pos_weight_max: float = 30.0,
        power: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.pos_weight_max = float(pos_weight_max)
        self.power = float(power)
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32).clamp(min=1.0, max=self.pos_weight_max)
            self.register_buffer("pos_weight", pw)
        else:
            self.register_buffer("pos_weight", None)

    def set_pos_weight_from_labels(self, train_labels) -> None:
        y = torch.as_tensor(train_labels, dtype=torch.float32)
        n_pos = y.sum(dim=0)
        n_neg = len(y) - n_pos
        pw = torch.pow((n_neg / n_pos.clamp(min=1.0)).clamp(min=1.0), self.power).clamp(
            min=1.0, max=self.pos_weight_max
        )
        self.pos_weight = pw

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        targets_eff = targets * (1.0 - self.label_smoothing) if self.label_smoothing > 0 else targets
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_eff, pos_weight=pw)


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


@register_loss("asymmetric")
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss with probability margin shifting (Ridnik / Ben-Baruch et al., ICCV 2021)."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        # Shift negative probability: 1 - max(p - clip, 0) = min(1 - p + clip, 1)
        probs_neg = 1.0 - probs_pos
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos)
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = probs_pos * targets + probs_neg * (1.0 - targets)
                gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
                focal_w = torch.pow((1.0 - pt).clamp(min=0.0), gamma)
            loss = loss * focal_w

        return -loss.mean()
