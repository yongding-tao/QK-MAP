"""Core QK-MAP modules.

QK-MAP maps the hidden state to bounded periodic coordinates before the
query and key projections.  The value path is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MagnitudeAwareProjection(nn.Module):
    """Learned periodic feature map with the same input and output width.

    For ``periodic_ratio=0.5``, the output is

        [cos(x W_p + b_p), sin(x W_p + b_p)]

    and has the same width as ``x``.  Smaller ratios retain an identity-like
    learned linear branch, which is useful for the ratio ablation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        *,
        periodic_ratio: float = 0.5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if not 0.0 <= periodic_ratio <= 0.5:
            raise ValueError("periodic_ratio must be in [0, 0.5]")

        periodic_dim = int(output_dim * periodic_ratio)
        linear_dim = output_dim - 2 * periodic_dim
        if periodic_ratio > 0.0 and periodic_dim == 0:
            raise ValueError("periodic_ratio is too small for this output width")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.periodic_ratio = periodic_ratio
        self.periodic_dim = periodic_dim
        self.linear_dim = linear_dim
        self.input_linear = nn.Linear(
            input_dim,
            periodic_dim + linear_dim,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_linear(x)
        periodic, linear = projected.split(
            (self.periodic_dim, self.linear_dim),
            dim=-1,
        )
        pieces = [torch.cos(periodic), torch.sin(periodic)]
        if self.linear_dim:
            pieces.append(linear)
        return torch.cat(pieces, dim=-1)


class QKMAPProjection(nn.Module):
    """Shared QK-MAP followed by separate Q, K, and V projections."""

    def __init__(
        self,
        hidden_size: int,
        query_size: int,
        key_size: int,
        value_size: int,
        *,
        periodic_ratio: float = 0.5,
        projection_bias: bool = False,
        map_bias: bool = True,
        residual: str = "none",
    ) -> None:
        super().__init__()
        if residual not in {"none", "add", "learnable", "learnable_per_dim"}:
            raise ValueError(f"Unsupported residual mode: {residual}")

        self.map = MagnitudeAwareProjection(
            hidden_size,
            hidden_size,
            periodic_ratio=periodic_ratio,
            bias=map_bias,
        )
        self.q_proj = nn.Linear(
            hidden_size, query_size, bias=projection_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, key_size, bias=projection_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, value_size, bias=projection_bias
        )
        self.residual = residual

        self.residual_alpha: nn.Parameter | None = None
        if residual == "learnable":
            self.residual_alpha = nn.Parameter(torch.zeros(1))
        elif residual == "learnable_per_dim":
            self.residual_alpha = nn.Parameter(torch.zeros(hidden_size))

    def mapped_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mapped = self.map(hidden_states)
        if self.residual == "none":
            return mapped
        if self.residual == "add":
            return mapped + hidden_states
        if self.residual_alpha is None:
            raise RuntimeError("Missing residual parameter")
        return mapped + hidden_states * self.residual_alpha.to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mapped = self.mapped_states(hidden_states)
        query = self.q_proj(mapped)
        key = self.k_proj(mapped)
        value = self.v_proj(hidden_states)
        return query, key, value


@dataclass(frozen=True)
class QKMAPConfig:
    """Small serializable configuration shared by training scripts."""

    periodic_ratio: float = 0.5
    map_bias: bool = True
    residual: str = "none"

