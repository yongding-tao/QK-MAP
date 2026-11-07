import pandas as pd
import torch

from config import (
    KEY_POS,
    METHODS,
    N_SEEDS,
    OUTPUT_PATH,
    QUERY_POS,
    SEED_BASE,
    ensure_output_dirs,
    validate_config,
)
from data_gen import exp2_data_generator, gamma_grid_tensor, make_exp2_sequence_batch, response_window_mask, sample_unit_direction
from metrics import monotonic_violation_rate_torch, s_max_torch, safe_spearman_torch
from operators import build_all_operators
from plot_results import plot_exp2_logit_vs_gamma, plot_exp2_response_spearman_box, plot_exp2_smax_box


def _should_report(index: int, total: int, interval: int) -> bool:
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def run_experiment_2() -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_config()
    ensure_output_dirs()

    gammas = gamma_grid_tensor()
    response_mask = response_window_mask(gammas)
    response_gammas = gammas[response_mask]
    curve_records: list[dict[str, float | int | str]] = []
    per_seed_records: list[dict[str, float | int | str]] = []

    print(f"[exp2] start: seeds={N_SEEDS}, gamma_count={gammas.numel()}", flush=True)

    with torch.no_grad():
        for seed_offset in range(N_SEEDS):
            seed = SEED_BASE + seed_offset
            if _should_report(seed_offset, N_SEEDS, interval=10):
                print(f"[exp2] running seed {seed_offset + 1}/{N_SEEDS} (seed={seed})", flush=True)
            data_generator = exp2_data_generator(seed)
            direction = sample_unit_direction(data_generator)
            sequence_batch = make_exp2_sequence_batch(direction, gammas)
            operators = build_all_operators(seed)

            gamma_values = gammas.cpu().tolist()
            for method, operator in operators.items():
                logits = operator(sequence_batch, output="logit", query_pos=QUERY_POS, key_pos=KEY_POS)
                response_logits = logits[response_mask]
                logits_cpu = logits.cpu().tolist()

                curve_records.extend(
                    {
                        "method": method,
                        "seed": seed,
                        "gamma": float(gamma_value),
                        "S_01": float(logit_value),
                    }
                    for gamma_value, logit_value in zip(gamma_values, logits_cpu)
                )
                per_seed_records.append(
                    {
                        "method": method,
                        "seed": seed,
                        "S_max": float(s_max_torch(logits).item()),
                        "resp_spearman": float(safe_spearman_torch(response_gammas, response_logits).item()),
                        "mono_violation_rate": float(monotonic_violation_rate_torch(response_logits).item()),
                    }
                )

    curves_df = pd.DataFrame(curve_records)
    per_seed_df = pd.DataFrame(per_seed_records)
    summary_df = (
        per_seed_df.groupby("method", sort=False)
        .agg(
            mean_S_max=("S_max", "mean"),
            std_S_max=("S_max", "std"),
            mean_resp_spearman=("resp_spearman", "mean"),
            std_resp_spearman=("resp_spearman", "std"),
            median_resp_spearman=("resp_spearman", "median"),
            q25_resp_spearman=("resp_spearman", lambda values: values.quantile(0.25)),
            q75_resp_spearman=("resp_spearman", lambda values: values.quantile(0.75)),
            mean_mono_violation_rate=("mono_violation_rate", "mean"),
            std_mono_violation_rate=("mono_violation_rate", "std"),
        )
        .reset_index()
    )

    curves_df.to_csv(OUTPUT_PATH / "exp2_curves.csv", index=False)
    per_seed_df.to_csv(OUTPUT_PATH / "exp2_per_seed.csv", index=False)
    summary_df.to_csv(OUTPUT_PATH / "exp2_summary.csv", index=False)

    plot_exp2_logit_vs_gamma()
    plot_exp2_smax_box()
    plot_exp2_response_spearman_box()
    print("[exp2] done", flush=True)

    return summary_df, per_seed_df


def main() -> None:
    summary_df, _ = run_experiment_2()
    print("exp2_summary.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
