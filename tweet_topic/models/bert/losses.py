"""Loss functions for Multi-Label Classification: BCE, Focal Loss, and Asymmetric Loss (ASL)."""

from __future__ import annotations

import torch
import torch.nn as nn
from tweet_topic.core.registry import register_loss


@register_loss("bce")
class StandardBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


@register_loss("focal")
class MultiLabelFocalLoss(nn.Module):
    """Focal Loss adapted for Multi-label classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()


@register_loss("asymmetric")
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for Multi-Label Classification (Ben-Baruch et al., 2021).
    
    Dynamically down-weights easy negatives to prevent gradient domination in sparse multi-label problems.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculating Probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1.0 - probs

        # Asymmetric Clipping for negatives
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        # Basic CE calculation
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = probs_pos * targets
            pt1 = probs_neg * (1.0 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()
