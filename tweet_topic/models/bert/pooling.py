"""Pooling and classification heads that sit on top of a token-level encoder.

Two ideas live here, each with a short statement of what it can and cannot do:

AttentionPooling
    Learns a score per token and takes the weighted average, instead of trusting a single ``<s>``/``[CLS]``
    vector. In a 28-word tweet the topical cue ("goal", "Billboard", "vaccine") can sit anywhere, so letting
    the model choose where to look is a reasonable inductive bias. It adds ~200k parameters; whether it beats
    plain CLS pooling on this dataset is an empirical question.

MultiSampleDropoutHead
    Applies several independent dropout masks to the *same* pooled vector and shares one linear layer.
    It is a **training-time** regulariser: with dropout off (eval mode) all branches are identical, so at
    inference it is exactly one linear layer. In training it returns the per-branch logits so that the loss
    can be averaged over branches (the formulation of Inoue, 2019) rather than over averaged logits.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Additive attention pooling.

    u_i = tanh(W h_i + b),  alpha_i = softmax_i(v^T u_i),  h_pool = sum_i alpha_i h_i
    """

    def __init__(self, hidden_size: int, attn_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(hidden_size, attn_dim)
        self.context = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.context(torch.tanh(self.proj(hidden))).squeeze(-1)  # [B, L]
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.to(scores.device) == 0, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        pooled = torch.sum(alpha.unsqueeze(-1) * hidden, dim=1)
        return pooled, alpha


class MultiSampleDropoutHead(nn.Module):
    """Shared linear layer evaluated under several independent dropout masks.

    Training mode returns logits of shape [S, B, C]; eval mode returns [B, C].
    """

    def __init__(self, in_dim: int, num_classes: int, num_samples: int = 5, dropout: Union[float, Sequence[float]] = 0.2):
        super().__init__()
        if isinstance(dropout, (int, float)):
            probs = [float(dropout)] * num_samples
        else:
            probs = [float(p) for p in dropout]
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in probs])
        self.fc = nn.Linear(in_dim, num_classes)

    @property
    def num_samples(self) -> int:
        return len(self.dropouts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            branches = torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0)
            return branches if self.num_samples > 1 else branches[0]
        return self.fc(x)


def reduce_multisample_loss(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Averages the loss over MSD branches ([S, B, C]); passes plain [B, C] logits straight through."""
    if logits.dim() == 3:
        return torch.stack([criterion(branch, labels) for branch in logits], dim=0).mean()
    return criterion(logits, labels)
