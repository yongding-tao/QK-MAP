import math

import torch
import torch.nn as nn

from config import (
    COSINE_TAU,
    D_MODEL,
    INIT_STD,
    METHOD_SEED_OFFSETS,
    METHODS,
    RMSNORM_EPS,
    RUNTIME_DEVICE,
)


def make_generator(seed: int) -> torch.Generator:
    generator_device = "cpu" if RUNTIME_DEVICE.type == "cpu" else "cuda"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def l2_normalize(x: torch.Tensor, eps: float = RMSNORM_EPS) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / norm.clamp_min(eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMSNORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=RUNTIME_DEVICE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class AttentionOperator(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.q_weight = nn.Parameter(torch.empty(D_MODEL, D_MODEL, device=RUNTIME_DEVICE))
        self.k_weight = nn.Parameter(torch.empty(D_MODEL, D_MODEL, device=RUNTIME_DEVICE))
        self.q_rmsnorm = RMSNorm(D_MODEL) if method == "qkrn" else None
        self.k_rmsnorm = RMSNorm(D_MODEL) if method == "qkrn" else None
        self.qkmap_weight = (
            nn.Parameter(torch.empty(D_MODEL // 2, D_MODEL, device=RUNTIME_DEVICE))
            if method == "qkmap"
            else None
        )

    def reset_parameters(self, seed: int) -> None:
        generator = make_generator(seed)
        with torch.no_grad():
            self.q_weight.normal_(mean=0.0, std=INIT_STD, generator=generator)
            self.k_weight.normal_(mean=0.0, std=INIT_STD, generator=generator)
            if self.qkmap_weight is not None:
                self.qkmap_weight.normal_(mean=0.0, std=INIT_STD, generator=generator)
            if self.q_rmsnorm is not None:
                self.q_rmsnorm.weight.fill_(1.0)
            if self.k_rmsnorm is not None:
                self.k_rmsnorm.weight.fill_(1.0)

    def _qkmap_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.qkmap_weight is None:
            raise NotImplementedError("QK-MAP transform is not implemented for this method")
        projected = x @ self.qkmap_weight.T
        return torch.cat((torch.cos(projected), torch.sin(projected)), dim=-1)

    def _project(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x @ weight.T

    def _apply_post_norm(self, tensor: torch.Tensor, branch: str) -> torch.Tensor:
        if self.method == "qkrn":
            norm_module = self.q_rmsnorm if branch == "q" else self.k_rmsnorm
            return norm_module(tensor)
        if self.method == "cosine":
            return l2_normalize(tensor)
        return tensor

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "qkmap":
            return self._qkmap_transform(x)
        return x

    def compute_q(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.preprocess_input(x)
        q = self._project(hidden_states, self.q_weight)
        return self._apply_post_norm(q, branch="q")

    def compute_k(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.preprocess_input(x)
        k = self._project(hidden_states, self.k_weight)
        return self._apply_post_norm(k, branch="k")

    def compute_qk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.preprocess_input(x)
        q = self._apply_post_norm(self._project(hidden_states, self.q_weight), branch="q")
        k = self._apply_post_norm(self._project(hidden_states, self.k_weight), branch="k")
        return q, k

    def compute_logit_tensor(self, x: torch.Tensor, query_pos: int, key_pos: int) -> torch.Tensor:
        q, k = self.compute_qk(x)
        logit = (q[..., query_pos, :] * k[..., key_pos, :]).sum(dim=-1) / math.sqrt(D_MODEL)
        if self.method == "cosine":
            logit = COSINE_TAU * logit
        return logit

    def compute_logit(self, x: torch.Tensor, query_pos: int, key_pos: int) -> float:
        logit = self.compute_logit_tensor(x, query_pos, key_pos)
        return float(logit.item())

    def forward(
        self,
        x: torch.Tensor,
        output: str = "q",
        query_pos: int = 0,
        key_pos: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if output == "q":
            return self.compute_q(x)
        if output == "k":
            return self.compute_k(x)
        if output == "qk":
            return self.compute_qk(x)
        if output == "logit":
            return self.compute_logit_tensor(x, query_pos, key_pos)
        raise ValueError(f"Unsupported output mode: {output}")


def build_operator(method: str, base_seed: int) -> nn.Module:
    operator = AttentionOperator(method).to(RUNTIME_DEVICE)
    operator.reset_parameters(base_seed + METHOD_SEED_OFFSETS[method])
    operator.eval()
    return operator


def build_all_operators(base_seed: int) -> dict[str, nn.Module]:
    return {method: build_operator(method, base_seed) for method in METHODS}
