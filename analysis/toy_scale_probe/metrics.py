import numpy as np
import torch


def _rankdata_torch(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=-1)
    ranks = torch.argsort(order, dim=-1).to(dtype=values.dtype)
    return ranks + 1.0


def safe_spearman_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rank_x = _rankdata_torch(x.to(dtype=torch.float64))
    rank_y = _rankdata_torch(y.to(dtype=torch.float64))
    centered_x = rank_x - rank_x.mean(dim=-1, keepdim=True)
    centered_y = rank_y - rank_y.mean(dim=-1, keepdim=True)
    covariance = (centered_x * centered_y).sum(dim=-1)
    denominator = torch.sqrt((centered_x.square().sum(dim=-1)) * (centered_y.square().sum(dim=-1)))
    rho = torch.where(denominator > 0, covariance / denominator, torch.zeros_like(covariance))
    return torch.nan_to_num(rho, nan=0.0)


def log_ratio_mae_torch(scales: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    error = torch.log(norms.clamp_min(1e-12)) - torch.log(scales.clamp_min(1e-12))
    sorted_error, _ = torch.sort(error.to(dtype=torch.float64), dim=-1)
    num_values = sorted_error.shape[-1]
    if num_values < 2:
        return torch.zeros(sorted_error.shape[:-1], dtype=sorted_error.dtype, device=sorted_error.device)
    coefficients = (2 * torch.arange(num_values, device=sorted_error.device, dtype=sorted_error.dtype)) - num_values + 1
    pairwise_abs_sum = (sorted_error * coefficients).sum(dim=-1)
    return 2.0 * pairwise_abs_sum / (num_values * (num_values - 1))


def log_linear_r2_torch(scales: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    x = torch.log(scales.clamp_min(1e-12)).to(dtype=torch.float64)
    y = torch.log(norms.clamp_min(1e-12)).to(dtype=torch.float64)
    centered_x = x - x.mean(dim=-1, keepdim=True)
    centered_y = y - y.mean(dim=-1, keepdim=True)
    ss_x = centered_x.square().sum(dim=-1)
    ss_y = centered_y.square().sum(dim=-1)
    covariance = (centered_x * centered_y).sum(dim=-1)
    denom = ss_x * ss_y
    r2 = torch.where(denom > 0, covariance.square() / denom, torch.zeros_like(covariance))
    return torch.nan_to_num(r2, nan=0.0)


def s_max_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(values), dim=-1).values


def monotonic_violation_rate_torch(values: torch.Tensor) -> torch.Tensor:
    num_values = values.shape[-1]
    if num_values < 2:
        return torch.tensor(0.0, dtype=torch.float64, device=values.device)

    pairwise_diffs = values.unsqueeze(-1) - values.unsqueeze(-2)
    violations = torch.triu(pairwise_diffs > 0, diagonal=1).sum()
    total_pairs = num_values * (num_values - 1) // 2
    return violations.to(dtype=torch.float64) / float(total_pairs)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_tensor = torch.as_tensor(x, dtype=torch.float64)
    y_tensor = torch.as_tensor(y, dtype=torch.float64)
    return float(safe_spearman_torch(x_tensor, y_tensor).item())


def log_ratio_mae(scales: np.ndarray, norms: np.ndarray) -> float:
    scales_tensor = torch.as_tensor(scales, dtype=torch.float64)
    norms_tensor = torch.as_tensor(norms, dtype=torch.float64)
    return float(log_ratio_mae_torch(scales_tensor, norms_tensor).item())


def log_linear_r2(scales: np.ndarray, norms: np.ndarray) -> float:
    scales_tensor = torch.as_tensor(scales, dtype=torch.float64)
    norms_tensor = torch.as_tensor(norms, dtype=torch.float64)
    return float(log_linear_r2_torch(scales_tensor, norms_tensor).item())


def s_max(values: np.ndarray) -> float:
    values_tensor = torch.as_tensor(values, dtype=torch.float64)
    return float(s_max_torch(values_tensor).item())


def response_slice(gammas: np.ndarray, values: np.ndarray, gamma_min: float, gamma_max: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (gammas >= gamma_min) & (gammas <= gamma_max)
    return gammas[mask], values[mask]


def monotonic_violation_rate(values: np.ndarray) -> float:
    values_tensor = torch.as_tensor(values, dtype=torch.float64)
    return float(monotonic_violation_rate_torch(values_tensor).item())
