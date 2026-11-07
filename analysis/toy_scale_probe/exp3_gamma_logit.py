import numpy as np
import pandas as pd
import torch

from config import (
    KEY_POS,
    METHODS,
    N_SEEDS,
    QUERY_POS,
    SEED_BASE,
    ensure_output_dirs,
    validate_config,
)
from data_gen import (
    exp3_data_generator,
    gamma_grid_tensor,
    make_exp3_sequence_batch,
    response_window_mask,
    sample_orthogonal_unit_direction,
    sample_unit_direction,
    theta_grid_tensor,
)
from metrics import monotonic_violation_rate_torch, s_max_torch, safe_spearman_torch
from operators import build_all_operators
from plot_results import (
    plot_exp3_logit_vs_gamma_from_arrays,
    plot_exp3_response_spearman_box_from_arrays,
    plot_exp3_smax_box_from_arrays,
)


def _should_report(index: int, total: int, interval: int) -> bool:
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def run_experiment_3() -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_config()
    ensure_output_dirs()

    gammas = gamma_grid_tensor()
    thetas = theta_grid_tensor()
    response_mask = response_window_mask(gammas)
    response_gammas = gammas[response_mask]
    num_methods = len(METHODS)
    num_theta = int(thetas.numel())
    num_gamma = int(gammas.numel())
    num_pairs = N_SEEDS * num_theta
    curve_values = np.empty((num_methods, num_pairs, num_gamma), dtype=np.float32)
    smax_store = np.empty((num_methods, num_pairs), dtype=np.float32)
    resp_spearman_store = np.empty((num_methods, num_pairs), dtype=np.float32)
    mono_violation_store = np.empty((num_methods, num_pairs), dtype=np.float32)
    gamma_values = gammas.cpu().numpy()

    print(
        f"[exp3] start: seeds={N_SEEDS}, theta_count={thetas.numel()}, gamma_count={gammas.numel()}",
        flush=True,
    )

    with torch.no_grad():
        for seed_offset in range(N_SEEDS):
            seed = SEED_BASE + seed_offset
            if _should_report(seed_offset, N_SEEDS, interval=10):
                print(f"[exp3] running seed {seed_offset + 1}/{N_SEEDS} (seed={seed})", flush=True)
            data_generator = exp3_data_generator(seed)
            query_direction = sample_unit_direction(data_generator)
            orthogonal_direction = sample_orthogonal_unit_direction(query_direction, data_generator)
            theta_radians = torch.deg2rad(thetas)
            key_directions = (
                torch.cos(theta_radians).unsqueeze(-1) * query_direction.unsqueeze(0)
                + torch.sin(theta_radians).unsqueeze(-1) * orthogonal_direction.unsqueeze(0)
            )
            sequence_batch = make_exp3_sequence_batch(query_direction, key_directions, gammas)
            flat_sequence_batch = sequence_batch.reshape(-1, sequence_batch.shape[-2], sequence_batch.shape[-1])
            operators = build_all_operators(seed)
            pair_start = seed_offset * num_theta
            pair_stop = pair_start + num_theta
            for method_idx, (_, operator) in enumerate(operators.items()):
                logits = operator(flat_sequence_batch, output="logit", query_pos=QUERY_POS, key_pos=KEY_POS)
                logits_by_theta = logits.reshape(thetas.shape[0], gammas.shape[0])
                response_logits_by_theta = logits_by_theta[:, response_mask]
                smax_tensor = s_max_torch(logits_by_theta)
                resp_spearman_tensor = safe_spearman_torch(
                    response_gammas.unsqueeze(0).expand(logits_by_theta.shape[0], -1),
                    response_logits_by_theta,
                )
                mono_violation_array = [
                    float(monotonic_violation_rate_torch(response_logits_by_theta[theta_idx]).item())
                    for theta_idx in range(logits_by_theta.shape[0])
                ]
                curve_values[method_idx, pair_start:pair_stop, :] = logits_by_theta.cpu().numpy().astype(np.float32, copy=False)
                smax_store[method_idx, pair_start:pair_stop] = smax_tensor.cpu().numpy().astype(np.float32, copy=False)
                resp_spearman_store[method_idx, pair_start:pair_stop] = (
                    resp_spearman_tensor.cpu().numpy().astype(np.float32, copy=False)
                )
                mono_violation_store[method_idx, pair_start:pair_stop] = np.asarray(
                    mono_violation_array,
                    dtype=np.float32,
                )

    grouped_curves = {}
    summary_records: list[dict[str, float | str]] = []
    for method_idx, method in enumerate(METHODS):
        method_curves = curve_values[method_idx]
        method_smax = smax_store[method_idx]
        method_resp_spearman = resp_spearman_store[method_idx]
        method_mono_violation = mono_violation_store[method_idx]
        grouped_curves[method] = {
            "min_value": np.min(method_curves, axis=0),
            "q25": np.quantile(method_curves, 0.25, axis=0),
            "median": np.quantile(method_curves, 0.5, axis=0),
            "q75": np.quantile(method_curves, 0.75, axis=0),
            "max_value": np.max(method_curves, axis=0),
        }
        summary_records.append(
            {
                "method": method,
                "mean_S_max": float(np.mean(method_smax)),
                "std_S_max": float(np.std(method_smax, ddof=1)),
                "mean_resp_spearman": float(np.mean(method_resp_spearman)),
                "std_resp_spearman": float(np.std(method_resp_spearman, ddof=1)),
                "median_resp_spearman": float(np.median(method_resp_spearman)),
                "q25_resp_spearman": float(np.quantile(method_resp_spearman, 0.25)),
                "q75_resp_spearman": float(np.quantile(method_resp_spearman, 0.75)),
                "mean_mono_violation_rate": float(np.mean(method_mono_violation)),
                "std_mono_violation_rate": float(np.std(method_mono_violation, ddof=1)),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    plot_exp3_logit_vs_gamma_from_arrays(gamma_values, grouped_curves)
    plot_exp3_smax_box_from_arrays(
        {method: smax_store[method_idx] for method_idx, method in enumerate(METHODS)}
    )
    plot_exp3_response_spearman_box_from_arrays(
        {method: resp_spearman_store[method_idx] for method_idx, method in enumerate(METHODS)}
    )
    print("[exp3] done", flush=True)

    return summary_df, pd.DataFrame()


def main() -> None:
    summary_df, _ = run_experiment_3()
    print("exp3_summary (in memory only)")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
