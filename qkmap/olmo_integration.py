"""Small helpers for inserting QK-MAP into an OLMo-style attention block.

OLMo-core keeps the full attention forward pass in its attention module.  The
helpers below isolate the only changed part: Q and K consume the shared mapped
state, while V continues to consume the original hidden state.
"""

from __future__ import annotations

import torch
from torch import nn

from .core import MagnitudeAwareProjection


def build_qkmap(
    hidden_size: int,
    *,
    periodic_ratio: float = 0.5,
    residual: str = "none",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, nn.Parameter | None]:
    """Build the shared feature map and the optional residual parameter."""

    if residual not in {"none", "add", "learnable", "learnable_per_dim"}:
        raise ValueError(f"Unsupported residual mode: {residual}")
    layer = MagnitudeAwareProjection(
        hidden_size,
        hidden_size,
        periodic_ratio=periodic_ratio,
    ).to(dtype=dtype, device=device)

    alpha: nn.Parameter | None = None
    if residual == "learnable":
        alpha = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
    elif residual == "learnable_per_dim":
        alpha = nn.Parameter(
            torch.zeros(hidden_size, dtype=dtype, device=device)
        )
    return layer, alpha


def mapped_qk_input(
    hidden_states: torch.Tensor,
    qkmap: nn.Module,
    *,
    residual: str = "none",
    residual_alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the tensor consumed by the Q and K projections."""

    mapped = qkmap(hidden_states)
    if residual == "none":
        return mapped
    if residual == "add":
        return mapped + hidden_states
    if residual in {"learnable", "learnable_per_dim"}:
        if residual_alpha is None:
            raise ValueError("residual_alpha is required for a learnable residual")
        return mapped + hidden_states * residual_alpha.to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    raise ValueError(f"Unsupported residual mode: {residual}")


def project_qkv(
    hidden_states: torch.Tensor,
    *,
    qkmap: nn.Module,
    q_proj: nn.Module,
    k_proj: nn.Module,
    v_proj: nn.Module,
    residual: str = "none",
    residual_alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q/K from QK-MAP features and V from the original state."""

    qk_input = mapped_qk_input(
        hidden_states,
        qkmap,
        residual=residual,
        residual_alpha=residual_alpha,
    )
    return (
        q_proj(qk_input),
        k_proj(qk_input),
        v_proj(hidden_states),
    )

