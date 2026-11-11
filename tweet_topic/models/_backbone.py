"""Shared helpers for turning a (frozen or adapted) language-model backbone into a feature extractor.

Both the Liquid d1 probe and the Qwen QLoRA classifier need the same three things:
1. Pull a hidden-state tensor out of whatever output object the backbone returns.
2. Pool token states into one vector per tweet.
3. Know the *real* hidden width (VL / multimodal configs often nest it under ``text_config``).
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def extract_hidden(outputs: Any) -> torch.Tensor:
    """Returns the last-layer token states [B, L, H] from a model output.

    Never falls back to ``outputs.logits``: those are vocabulary-sized scores, not features, and feeding
    them to a ``Linear(hidden, C)`` head is either a shape error or a silent bug.
    """
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    last = getattr(outputs, "last_hidden_state", None)
    if last is not None:
        return last
    raise RuntimeError(
        f"{type(outputs).__name__} exposes neither hidden_states nor last_hidden_state. "
        "Call the backbone with output_hidden_states=True."
    )


def pool_tokens(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor], mode: str = "mean") -> torch.Tensor:
    """Pools [B, L, H] token states into [B, H].

    mean : padding-aware average over tokens (robust default for decoder-only models).
    last : state of the last non-padding token (works for right padding and left padding).
    """
    if mode == "last":
        if attention_mask is None:
            return hidden[:, -1, :]
        seq_len = hidden.size(1)
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
        last_idx = (attention_mask.to(hidden.device).long() * (positions + 1)).argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]

    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def infer_hidden_size(model: Any, sample_inputs: Optional[dict] = None) -> int:
    """Determines the hidden width.

    Prefers a real forward pass over config guessing, because composite (vision-language) configs keep the
    text width under ``text_config`` and a wrong default would silently build a mismatched head.
    """
    if sample_inputs is not None:
        with torch.no_grad():
            out = model(**sample_inputs, output_hidden_states=True)
        return int(extract_hidden(out).size(-1))

    cfg = getattr(model, "config", None)
    for holder in (getattr(cfg, "text_config", None), cfg):
        if holder is None:
            continue
        for key in ("hidden_size", "d_model", "dim"):
            value = getattr(holder, key, None)
            if value:
                return int(value)
    raise RuntimeError("Cannot infer hidden size; pass sample_inputs so a real forward pass can be used.")


def load_backbone(
    model_name: str,
    dtype: torch.dtype,
    device_map: Any = None,
    prefer: str = "causal",
    **kwargs: Any,
) -> Any:
    """Loads a backbone, trying one Auto class first and falling back to the other.

    prefer="causal": needs LM logits (zero-shot scoring). prefer="base": needs hidden states only (probes,
    QLoRA classifiers), which avoids materialising the vocabulary-sized LM head when a base model exists.
    Some hub repos (e.g. the multimodal Liquid d1) register only one of these classes, so a single hard-coded
    Auto class tends to break across transformers versions.
    """
    from transformers import AutoModel, AutoModelForCausalLM

    order = (AutoModelForCausalLM, AutoModel) if prefer == "causal" else (AutoModel, AutoModelForCausalLM)
    errors = []
    for auto_cls in order:
        try:
            return auto_cls.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - we want to try the next class on any failure
            errors.append(f"{auto_cls.__name__}: {exc}")
    raise RuntimeError("Could not load backbone '%s'.\n%s" % (model_name, "\n".join(errors)))
