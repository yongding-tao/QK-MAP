import math

import torch

from config import (
    D_MODEL,
    EXP1_INPUT_SEED,
    GAMMA_MAX,
    GAMMA_MIN,
    GAMMA_RESP_MAX,
    GAMMA_RESP_MIN,
    N_GAMMA,
    N_THETA,
    RUNTIME_DEVICE,
    S0_FIXED,
    S1_BASE,
    THETA_MAX_DEG,
    THETA_MIN_DEG,
    exp1_scale_grid,
)
from operators import make_generator


def sample_unit_direction(generator: torch.Generator) -> torch.Tensor:
    vector = torch.randn(D_MODEL, generator=generator, device=RUNTIME_DEVICE)
    return vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)


def sample_unit_directions(num_samples: int, generator: torch.Generator) -> torch.Tensor:
    vectors = torch.randn(num_samples, D_MODEL, generator=generator, device=RUNTIME_DEVICE)
    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)


def sample_orthogonal_unit_direction(
    base_direction: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    while True:
        candidate = sample_unit_direction(generator)
        orthogonal = candidate - torch.dot(candidate, base_direction) * base_direction
        norm = torch.linalg.vector_norm(orthogonal)
        if float(norm.item()) > 1e-8:
            return orthogonal / norm


def rotate_direction(
    base_direction: torch.Tensor,
    orthogonal_direction: torch.Tensor,
    angle_deg: float,
) -> torch.Tensor:
    angle_rad = math.radians(float(angle_deg))
    return math.cos(angle_rad) * base_direction + math.sin(angle_rad) * orthogonal_direction


def build_scaled_samples(scales: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    if scales.ndim != 1:
        raise ValueError("scales must be a 1D tensor")
    if directions.ndim == 1:
        return scales.unsqueeze(-1) * directions.unsqueeze(0)
    if directions.ndim == 2 and directions.shape[0] == scales.shape[0]:
        return scales.unsqueeze(-1) * directions
    raise ValueError("directions must be a single direction or one direction per scale")


def exp1_scale_grid_tensor(method: str | None = None) -> torch.Tensor:
    return torch.as_tensor(exp1_scale_grid(method), dtype=torch.float32, device=RUNTIME_DEVICE)


def make_exp1_colinear_samples(
    generator: torch.Generator,
    method: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = exp1_scale_grid_tensor(method)
    direction = sample_unit_direction(generator)
    samples = build_scaled_samples(scales, direction)
    return direction, scales, samples


def make_exp1_independent_samples(
    generator: torch.Generator,
    method: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = exp1_scale_grid_tensor(method)
    directions = sample_unit_directions(scales.shape[0], generator)
    samples = build_scaled_samples(scales, directions)
    return directions, scales, samples


def build_colinear_sequence(scales: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    if scales.ndim == 1 and direction.ndim == 1:
        return scales.unsqueeze(-1) * direction.unsqueeze(0)
    if scales.ndim == 2 and direction.ndim == 2:
        return scales.unsqueeze(-1) * direction.unsqueeze(-2)
    raise ValueError("Unsupported shapes for build_colinear_sequence")


def make_exp2_sequence(direction: torch.Tensor, gamma: float) -> torch.Tensor:
    scales = torch.tensor([S0_FIXED, float(gamma) * S1_BASE], dtype=torch.float32, device=RUNTIME_DEVICE)
    return build_colinear_sequence(scales, direction)


def make_exp2_sequence_batch(direction: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    scales = torch.stack(
        (
            torch.full_like(gammas, float(S0_FIXED)),
            gammas.to(dtype=torch.float32) * float(S1_BASE),
        ),
        dim=-1,
    )
    return build_colinear_sequence(scales, direction.unsqueeze(0).expand(gammas.shape[0], -1))


def make_exp3_sequence(query_direction: torch.Tensor, key_direction: torch.Tensor, gamma: float) -> torch.Tensor:
    query = torch.tensor(float(S0_FIXED), dtype=torch.float32, device=RUNTIME_DEVICE) * query_direction
    key = torch.tensor(float(gamma) * S1_BASE, dtype=torch.float32, device=RUNTIME_DEVICE) * key_direction
    return torch.stack((query, key), dim=0)


def make_exp3_sequence_batch(
    query_direction: torch.Tensor,
    key_directions: torch.Tensor,
    gammas: torch.Tensor,
) -> torch.Tensor:
    num_theta = key_directions.shape[0]
    num_gamma = gammas.shape[0]
    query = (float(S0_FIXED) * query_direction).view(1, 1, -1).expand(num_theta, num_gamma, -1)
    key = key_directions[:, None, :] * (gammas.view(1, num_gamma, 1).to(dtype=torch.float32) * float(S1_BASE))
    return torch.stack((query, key), dim=-2)


def exp1_data_generator(seed: int) -> torch.Generator:
    return make_generator(seed + EXP1_INPUT_SEED + 10_000)


def exp2_data_generator(seed: int) -> torch.Generator:
    return make_generator(seed + 20_000)


def exp3_data_generator(seed: int) -> torch.Generator:
    return make_generator(seed + 30_000)


def gamma_grid_tensor() -> torch.Tensor:
    return torch.linspace(GAMMA_MIN, GAMMA_MAX, N_GAMMA, device=RUNTIME_DEVICE)


def theta_grid_tensor() -> torch.Tensor:
    return torch.linspace(THETA_MIN_DEG, THETA_MAX_DEG, N_THETA, device=RUNTIME_DEVICE)


def response_window_mask(gammas: torch.Tensor) -> torch.Tensor:
    return (gammas >= GAMMA_RESP_MIN) & (gammas <= GAMMA_RESP_MAX)
