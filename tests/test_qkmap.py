from __future__ import annotations

import torch

from qkmap import (
    MagnitudeAwareProjection,
    QKMAPProjection,
    build_qkmap,
    project_qkv,
)


def test_main_map_has_fixed_feature_norm() -> None:
    torch.manual_seed(0)
    hidden_size = 16
    layer = MagnitudeAwareProjection(
        hidden_size,
        periodic_ratio=0.5,
    )
    x = torch.randn(3, 5, hidden_size)
    mapped = layer(x)
    expected = torch.full((3, 5), hidden_size / 2)
    assert mapped.shape == x.shape
    assert torch.allclose(mapped.square().sum(dim=-1), expected, atol=1e-5)


def test_magnitude_changes_periodic_coordinates() -> None:
    layer = MagnitudeAwareProjection(4, periodic_ratio=0.5, bias=False)
    with torch.no_grad():
        layer.input_linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
        )
    direction = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    assert not torch.allclose(layer(direction), layer(2.0 * direction))


def test_value_path_is_unchanged_by_qk_map() -> None:
    torch.manual_seed(1)
    module = QKMAPProjection(8, 8, 8, 8, periodic_ratio=0.5)
    x = torch.randn(2, 4, 8)
    _, _, value = module(x)
    assert torch.allclose(value, module.v_proj(x))


def test_olmo_helper_keeps_value_input_unchanged() -> None:
    torch.manual_seed(2)
    qkmap, alpha = build_qkmap(
        8,
        residual="learnable_per_dim",
    )
    q_proj = torch.nn.Linear(8, 8, bias=False)
    k_proj = torch.nn.Linear(8, 8, bias=False)
    v_proj = torch.nn.Linear(8, 8, bias=False)
    x = torch.randn(2, 3, 8)
    _, _, value = project_qkv(
        x,
        qkmap=qkmap,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        residual="learnable_per_dim",
        residual_alpha=alpha,
    )
    assert torch.allclose(value, v_proj(x))
