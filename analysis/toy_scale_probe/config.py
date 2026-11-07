from pathlib import Path
import math
import re

import numpy as np
import torch

# --- 维度与序列 ---
D_MODEL = 128  # d，必须为偶数（QK-MAP 需要）
SEQ_LEN_EXP2 = 2  # 实验 2/3 序列长度 L（固定为 2）

# --- 实验 1：单次散点图实验 ---
EXP1_OPERATOR_SEED = 2
EXP1_INPUT_SEED = 1
EXP1_SCALE_MIN = 0.0
EXP1_SCALE_MAX = 1500.0
EXP1_SCALE_STEP = 0.1
EXP1_QKRN_DENSE_MAX = 0.5
EXP1_QKRN_SCALE_STEP_DENSE = 0.001
EXP1_BATCH_SIZE = 128000  # 会自动向下对齐到总样本数的最大因子

# --- 实验 1：范数桶后处理分析 ---
EXP1_BUCKET_MODE = "target_q_norm_qkmap_cap"  # "auto_max_span_same_norm" | "target_q_norm_qkmap_cap"

EXP1_BUCKET_QUANTILE_LOW = 0.05
EXP1_BUCKET_QUANTILE_HIGH = 0.95
EXP1_BUCKET_NUM_CANDIDATE_CENTERS = 25
EXP1_BUCKET_RELATIVE_WIDTH = 0.05
EXP1_BUCKET_MIN_WIDTH = 1e-4
EXP1_BUCKET_MAX_EXPANSIONS = 6
EXP1_BUCKET_MIN_SAMPLES = 96
EXP1_BUCKET_VIS_SAMPLE_COUNT = 96
EXP1_BUCKET_PAIRWISE_SAMPLE_COUNT = 128
EXP1_BUCKET_PCA_SAMPLE_COUNT = 128
EXP1_BUCKET_TSNE_SAMPLE_COUNT = 128
EXP1_BUCKET_TOPK_DIMS = 12
EXP1_BUCKET_KDE_GRID_SIZE = 256
EXP1_TSNE_PERPLEXITY = 30.0
EXP1_TSNE_LEARNING_RATE = 200.0
EXP1_TSNE_MAX_ITER = 1000

EXP1_TARGET_Q_NORM_BY_METHOD = {
    "std": None,
    "qkrn": None,
    "cosine": None,
    # "qkmap": 1.87,
    "qkmap": None,
}
EXP1_TARGET_Q_NORM_RELATIVE_TOL = 0.001
EXP1_TARGET_Q_NORM_MIN_TOL = 1e-4
EXP1_TARGET_Q_NORM_MAX_EXPANSIONS = 6
EXP1_TARGET_Q_NORM_CAP_METHOD = "qkmap"

# --- 实验 2/3：重复实验次数 ---
N_SEEDS = 3000
SEED_BASE = 0  # 第 s 个种子为 SEED_BASE + s（s = 0, ..., N_SEEDS-1）

# --- 实验 2/3：gamma 扫描（确定性格点，非随机分布采样）---
GAMMA_MIN = 0.01
GAMMA_MAX = 3000.0
N_GAMMA = 1500
S0_FIXED = 1.0  # 位置 0（query）固定尺度
S1_BASE = 1.0  # 位置 1（key）基准尺度

# --- 实验 2/3 响应 Spearman 的中等 gamma 区间（闭区间，用于标量汇总）---
GAMMA_RESP_MIN = 0.316227766  # 10^{-0.5}
GAMMA_RESP_MAX = 200.0

# --- 实验 3：query/key 方向夹角扫描（确定性格点，单位：度）---
THETA_MIN_DEG = 0.0
THETA_MAX_DEG = 180.0
N_THETA = 181

# --- 实验 2/3 位置（固定，不作为扫描维度）---
QUERY_POS = 0
KEY_POS = 1

# --- 权重初始化 ---
INIT_STD = 0.02

# --- 输出路径 ---
OUTPUT_DIR = "./output"
FIG_DIR = OUTPUT_DIR

# --- 运行设备与算子 ---
DEVICE = "gpu"  # "cpu" | "gpu" | "cuda0 cuda1 cuda7"
COSINE_TAU = math.sqrt(D_MODEL)
RMSNORM_EPS = 1e-6

METHODS = ("std", "qkrn", "cosine", "qkmap")
METHOD_DISPLAY_NAMES = {
    "std": "STD",
    "qkrn": "QK-RMSNorm",
    "cosine": "QK-L2Norm",
    "qkmap": "QK-MAP",
}
METHOD_SEED_OFFSETS = {
    "std": 11,
    "qkrn": 23,
    "cosine": 37,
    "qkmap": 53,
}

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = (ROOT_DIR / OUTPUT_DIR).resolve()
FIG_PATH = (ROOT_DIR / FIG_DIR).resolve()


def _parse_cuda_token(token: str) -> int | None:
    match = re.fullmatch(r"cuda(?::)?(\d+)", token.strip().lower())
    if match is None:
        return None
    return int(match.group(1))


def _resolve_runtime_device(device_spec: str) -> tuple[torch.device, list[int]]:
    normalized = device_spec.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu"), []

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    if normalized == "gpu":
        if cuda_count > 0:
            return torch.device("cuda:0"), [0]
        return torch.device("cpu"), []

    requested_ids: list[int] = []
    for token in normalized.split():
        device_id = _parse_cuda_token(token)
        if device_id is None:
            raise ValueError(
                f"Unsupported DEVICE token: {token!r}. Use 'cpu', 'gpu', or tokens like 'cuda0 cuda1'."
            )
        if device_id not in requested_ids:
            requested_ids.append(device_id)

    if not requested_ids:
        raise ValueError("DEVICE must be 'cpu', 'gpu', or a non-empty list like 'cuda0 cuda1'")

    if not cuda_available:
        return torch.device("cpu"), []

    valid_ids = [device_id for device_id in requested_ids if device_id < cuda_count]
    if not valid_ids:
        raise ValueError(
            f"None of the requested GPUs are available: {requested_ids}. "
            f"Current machine exposes cuda0~cuda{cuda_count - 1}."
        )
    first_device_id = valid_ids[0]
    return torch.device(f"cuda:{first_device_id}"), [first_device_id]


def _largest_divisor_not_exceeding(total: int, requested: int) -> int:
    if total <= 0:
        raise ValueError("total must be positive")
    if requested <= 0:
        raise ValueError("requested must be positive")
    if requested >= total:
        return total
    for candidate in range(requested, 0, -1):
        if total % candidate == 0:
            return candidate
    return 1


def _uniform_exp1_scale_grid() -> np.ndarray:
    scales = np.arange(
        EXP1_SCALE_MIN,
        EXP1_SCALE_MAX + 0.5 * EXP1_SCALE_STEP,
        EXP1_SCALE_STEP,
        dtype=np.float64,
    )
    return scales[scales <= EXP1_SCALE_MAX + 1e-12]


def _qkrn_exp1_scale_grid() -> np.ndarray:
    dense_upper = min(EXP1_QKRN_DENSE_MAX, EXP1_SCALE_MAX)
    dense_scales = np.arange(
        EXP1_SCALE_MIN,
        dense_upper + 0.5 * EXP1_QKRN_SCALE_STEP_DENSE,
        EXP1_QKRN_SCALE_STEP_DENSE,
        dtype=np.float64,
    )
    dense_scales = dense_scales[dense_scales <= dense_upper + 1e-12]

    if dense_upper >= EXP1_SCALE_MAX:
        return dense_scales

    coarse_start = dense_upper + EXP1_SCALE_STEP
    coarse_scales = np.arange(
        coarse_start,
        EXP1_SCALE_MAX + 0.5 * EXP1_SCALE_STEP,
        EXP1_SCALE_STEP,
        dtype=np.float64,
    )
    coarse_scales = coarse_scales[coarse_scales <= EXP1_SCALE_MAX + 1e-12]
    return np.concatenate((dense_scales, coarse_scales), axis=0)


def exp1_scale_grid(method: str | None = None) -> np.ndarray:
    if method == "qkrn":
        return _qkrn_exp1_scale_grid()
    return _uniform_exp1_scale_grid()


def exp1_num_samples(method: str | None = None) -> int:
    return int(exp1_scale_grid(method).size)


def exp1_effective_batch_size(method: str | None = None) -> int:
    return _largest_divisor_not_exceeding(exp1_num_samples(method), EXP1_BATCH_SIZE)


RUNTIME_DEVICE, PARALLEL_DEVICE_IDS = _resolve_runtime_device(DEVICE)
USE_GPU = RUNTIME_DEVICE.type == "cuda"


def validate_config() -> None:
    if D_MODEL <= 0:
        raise ValueError("D_MODEL must be positive")
    if D_MODEL % 2 != 0:
        raise ValueError("D_MODEL must be even for the QK-MAP operator")
    if SEQ_LEN_EXP2 != 2:
        raise ValueError("SEQ_LEN_EXP2 must be exactly 2 for experiments 2 and 3")
    if EXP1_OPERATOR_SEED < 0 or EXP1_INPUT_SEED < 0:
        raise ValueError("EXP1 seeds must be non-negative")
    if EXP1_BATCH_SIZE <= 0:
        raise ValueError("EXP1_BATCH_SIZE must be positive")
    if EXP1_SCALE_MIN < 0.0 or EXP1_SCALE_MAX <= EXP1_SCALE_MIN:
        raise ValueError("Require 0 <= EXP1_SCALE_MIN < EXP1_SCALE_MAX")
    if EXP1_SCALE_STEP <= 0.0:
        raise ValueError("EXP1_SCALE_STEP must be positive")
    if not (EXP1_SCALE_MIN < EXP1_QKRN_DENSE_MAX <= EXP1_SCALE_MAX):
        raise ValueError("Require EXP1_SCALE_MIN < EXP1_QKRN_DENSE_MAX <= EXP1_SCALE_MAX")
    if EXP1_QKRN_SCALE_STEP_DENSE <= 0.0:
        raise ValueError("EXP1_QKRN_SCALE_STEP_DENSE must be positive")
    if EXP1_QKRN_SCALE_STEP_DENSE > EXP1_SCALE_STEP:
        raise ValueError("EXP1_QKRN_SCALE_STEP_DENSE should not exceed EXP1_SCALE_STEP")
    if EXP1_BUCKET_MODE not in {"auto_max_span_same_norm", "target_q_norm_qkmap_cap"}:
        raise ValueError("Unsupported EXP1_BUCKET_MODE")
    if not (0.0 <= EXP1_BUCKET_QUANTILE_LOW < EXP1_BUCKET_QUANTILE_HIGH <= 1.0):
        raise ValueError("Bucket quantile range must satisfy 0 <= low < high <= 1")
    if EXP1_BUCKET_NUM_CANDIDATE_CENTERS < 3:
        raise ValueError("EXP1_BUCKET_NUM_CANDIDATE_CENTERS must be at least 3")
    if EXP1_BUCKET_RELATIVE_WIDTH <= 0.0:
        raise ValueError("EXP1_BUCKET_RELATIVE_WIDTH must be positive")
    if EXP1_BUCKET_MIN_WIDTH <= 0.0:
        raise ValueError("EXP1_BUCKET_MIN_WIDTH must be positive")
    if EXP1_BUCKET_MAX_EXPANSIONS <= 0:
        raise ValueError("EXP1_BUCKET_MAX_EXPANSIONS must be positive")
    if EXP1_BUCKET_MIN_SAMPLES < 8:
        raise ValueError("EXP1_BUCKET_MIN_SAMPLES must be at least 8")
    if EXP1_BUCKET_VIS_SAMPLE_COUNT < 8:
        raise ValueError("EXP1_BUCKET_VIS_SAMPLE_COUNT must be at least 8")
    if EXP1_BUCKET_PAIRWISE_SAMPLE_COUNT < 8:
        raise ValueError("EXP1_BUCKET_PAIRWISE_SAMPLE_COUNT must be at least 8")
    if EXP1_BUCKET_PCA_SAMPLE_COUNT < 8:
        raise ValueError("EXP1_BUCKET_PCA_SAMPLE_COUNT must be at least 8")
    if EXP1_BUCKET_TSNE_SAMPLE_COUNT < 8:
        raise ValueError("EXP1_BUCKET_TSNE_SAMPLE_COUNT must be at least 8")
    if EXP1_BUCKET_TOPK_DIMS <= 0:
        raise ValueError("EXP1_BUCKET_TOPK_DIMS must be positive")
    if EXP1_BUCKET_KDE_GRID_SIZE < 64:
        raise ValueError("EXP1_BUCKET_KDE_GRID_SIZE must be at least 64")
    if EXP1_TSNE_PERPLEXITY <= 0.0:
        raise ValueError("EXP1_TSNE_PERPLEXITY must be positive")
    if EXP1_TSNE_LEARNING_RATE <= 0.0:
        raise ValueError("EXP1_TSNE_LEARNING_RATE must be positive")
    if EXP1_TSNE_MAX_ITER < 250:
        raise ValueError("EXP1_TSNE_MAX_ITER must be at least 250")
    if set(EXP1_TARGET_Q_NORM_BY_METHOD) != set(METHODS):
        raise ValueError("EXP1_TARGET_Q_NORM_BY_METHOD must provide exactly one entry per method")
    for method, target_value in EXP1_TARGET_Q_NORM_BY_METHOD.items():
        if target_value is not None and target_value < 0.0:
            raise ValueError(f"EXP1 target q norm for {method} must be non-negative or None")
    if EXP1_TARGET_Q_NORM_RELATIVE_TOL <= 0.0:
        raise ValueError("EXP1_TARGET_Q_NORM_RELATIVE_TOL must be positive")
    if EXP1_TARGET_Q_NORM_MIN_TOL <= 0.0:
        raise ValueError("EXP1_TARGET_Q_NORM_MIN_TOL must be positive")
    if EXP1_TARGET_Q_NORM_MAX_EXPANSIONS <= 0:
        raise ValueError("EXP1_TARGET_Q_NORM_MAX_EXPANSIONS must be positive")
    if EXP1_TARGET_Q_NORM_CAP_METHOD not in METHODS:
        raise ValueError("EXP1_TARGET_Q_NORM_CAP_METHOD must be one of METHODS")
    for method in METHODS:
        if exp1_num_samples(method) <= 0:
            raise ValueError(f"Experiment 1 scale grid for {method} must contain at least one sample")
        if exp1_effective_batch_size(method) <= 0:
            raise ValueError(f"Resolved EXP1 batch size for {method} must be positive")
    if N_SEEDS <= 0:
        raise ValueError("N_SEEDS must be positive")
    if not (0.0 < GAMMA_MIN < GAMMA_MAX):
        raise ValueError("Require 0 < GAMMA_MIN < GAMMA_MAX")
    if N_GAMMA < 2:
        raise ValueError("N_GAMMA must be at least 2")
    if not (0.0 <= THETA_MIN_DEG < THETA_MAX_DEG <= 180.0):
        raise ValueError("Require 0 <= THETA_MIN_DEG < THETA_MAX_DEG <= 180")
    if N_THETA < 2:
        raise ValueError("N_THETA must be at least 2")
    if QUERY_POS != 0 or KEY_POS != 1:
        raise ValueError("Experiments 2/3 assume QUERY_POS=0 and KEY_POS=1")
    if not (GAMMA_MIN <= GAMMA_RESP_MIN <= GAMMA_RESP_MAX <= GAMMA_MAX):
        raise ValueError("Response gamma window must lie inside the full gamma range")


def ensure_output_dirs() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    if FIG_PATH != OUTPUT_PATH:
        FIG_PATH.mkdir(parents=True, exist_ok=True)


def gamma_grid() -> np.ndarray:
    return np.linspace(GAMMA_MIN, GAMMA_MAX, N_GAMMA, dtype=np.float64)


def theta_grid() -> np.ndarray:
    return np.linspace(THETA_MIN_DEG, THETA_MAX_DEG, N_THETA, dtype=np.float64)
