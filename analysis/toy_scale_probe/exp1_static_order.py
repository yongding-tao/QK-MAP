from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    EXP1_BUCKET_MODE,
    EXP1_BUCKET_MAX_EXPANSIONS,
    EXP1_BUCKET_MIN_SAMPLES,
    EXP1_BUCKET_MIN_WIDTH,
    EXP1_BUCKET_NUM_CANDIDATE_CENTERS,
    EXP1_BUCKET_PAIRWISE_SAMPLE_COUNT,
    EXP1_BUCKET_PCA_SAMPLE_COUNT,
    EXP1_BUCKET_QUANTILE_HIGH,
    EXP1_BUCKET_QUANTILE_LOW,
    EXP1_BUCKET_RELATIVE_WIDTH,
    EXP1_BUCKET_TSNE_SAMPLE_COUNT,
    EXP1_OPERATOR_SEED,
    EXP1_TARGET_Q_NORM_BY_METHOD,
    EXP1_TARGET_Q_NORM_CAP_METHOD,
    EXP1_TARGET_Q_NORM_MAX_EXPANSIONS,
    EXP1_TARGET_Q_NORM_MIN_TOL,
    EXP1_TARGET_Q_NORM_RELATIVE_TOL,
    METHODS,
    RUNTIME_DEVICE,
    ensure_output_dirs,
    exp1_effective_batch_size,
    exp1_num_samples,
    validate_config,
)
from data_gen import (
    build_scaled_samples,
    exp1_data_generator,
    exp1_scale_grid_tensor,
    sample_unit_direction,
    sample_unit_directions,
)
from operators import build_all_operators
from plot_results import (
    plot_exp1_bucket_dim_stats,
    plot_exp1_bucket_dim_mean_std,
    plot_exp1_bucket_dim_violin,
    plot_exp1_bucket_heatmap,
    plot_exp1_bucket_pairwise_kde,
    plot_exp1_bucket_pairwise_distances,
    plot_exp1_bucket_pca,
    plot_exp1_bucket_selection,
    plot_exp1_bucket_tsne,
    plot_exp1_scatter_from_records,
)


def _should_report(index: int, total: int, interval: int = 10) -> bool:
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def _candidate_bucket_centers(q_norms: np.ndarray) -> np.ndarray:
    q_low = float(np.quantile(q_norms, EXP1_BUCKET_QUANTILE_LOW))
    q_high = float(np.quantile(q_norms, EXP1_BUCKET_QUANTILE_HIGH))
    if q_high - q_low <= 1e-12:
        return np.asarray([float(np.median(q_norms))], dtype=np.float64)
    return np.linspace(
        q_low,
        q_high,
        EXP1_BUCKET_NUM_CANDIDATE_CENTERS,
        dtype=np.float64,
    )


def _base_bucket_width(q_norms: np.ndarray) -> float:
    q_iqr = float(np.quantile(q_norms, 0.75) - np.quantile(q_norms, 0.25))
    q_span = float(np.max(q_norms) - np.min(q_norms))
    reference_scale = max(q_iqr, q_span, 1e-8)
    return max(EXP1_BUCKET_MIN_WIDTH, EXP1_BUCKET_RELATIVE_WIDTH * reference_scale)


def _select_norm_bucket(
    setting_key: str,
    method: str,
    input_norms: np.ndarray,
    q_norms: np.ndarray,
) -> dict[str, float | int | str]:
    input_span_total = max(float(np.max(input_norms) - np.min(input_norms)), 1e-12)
    centers = _candidate_bucket_centers(q_norms)
    base_width = _base_bucket_width(q_norms)

    best_record: dict[str, float | int | str] | None = None
    best_score = -float("inf")
    fallback_record: dict[str, float | int | str] | None = None
    fallback_score = -float("inf")

    for expansion_idx in range(EXP1_BUCKET_MAX_EXPANSIONS):
        width = base_width * (2**expansion_idx)
        for center in centers:
            lower = float(center - 0.5 * width)
            upper = float(center + 0.5 * width)
            mask = (q_norms >= lower) & (q_norms <= upper)
            count = int(np.count_nonzero(mask))
            if count < 2:
                continue

            selected_inputs = input_norms[mask]
            selected_q_norms = q_norms[mask]
            input_span = float(np.max(selected_inputs) - np.min(selected_inputs))
            input_span_ratio = input_span / input_span_total
            count_score = np.log1p(count)
            width_penalty = np.log1p(width / max(base_width, 1e-12))
            tightness_penalty = float(np.std(selected_q_norms)) / max(width, 1e-12)
            score = count_score + 3.0 * input_span_ratio - 0.45 * width_penalty - 1.5 * tightness_penalty

            record = {
                "setting": setting_key,
                "method": method,
                "bucket_center": float(center),
                "bucket_width": float(width),
                "bucket_lower": lower,
                "bucket_upper": upper,
                "bucket_count": count,
                "bucket_input_min": float(np.min(selected_inputs)),
                "bucket_input_max": float(np.max(selected_inputs)),
                "bucket_input_span": input_span,
                "bucket_input_span_ratio": input_span_ratio,
                "bucket_q_norm_mean": float(np.mean(selected_q_norms)),
                "bucket_q_norm_std": float(np.std(selected_q_norms)),
                "bucket_q_norm_std_over_width": tightness_penalty,
                "bucket_expansion_step": expansion_idx,
                "bucket_score": float(score),
            }

            if score > fallback_score:
                fallback_record = record
                fallback_score = score
            if count >= EXP1_BUCKET_MIN_SAMPLES and score > best_score:
                best_record = record
                best_score = score

    chosen = best_record if best_record is not None else fallback_record
    if chosen is None:
        raise RuntimeError(f"Failed to select a bucket for setting={setting_key}, method={method}")
    chosen["bucket_strategy"] = "auto_max_span_same_norm"
    chosen["selection_subsample_strategy"] = "bucket_range_mask"
    chosen["target_q_norm"] = float(chosen["bucket_center"])
    chosen["selection_cap_count"] = int(chosen["bucket_count"])
    chosen["selected_indices"] = np.flatnonzero((q_norms >= chosen["bucket_lower"]) & (q_norms <= chosen["bucket_upper"]))
    return chosen


def _target_mode_base_tol(q_norms: np.ndarray) -> float:
    q_iqr = float(np.quantile(q_norms, 0.75) - np.quantile(q_norms, 0.25))
    q_span = float(np.max(q_norms) - np.min(q_norms))
    reference_scale = max(q_iqr, q_span, 1e-8)
    return max(EXP1_TARGET_Q_NORM_MIN_TOL, EXP1_TARGET_Q_NORM_RELATIVE_TOL * reference_scale)


def _resolve_target_q_norm(method: str, q_norms: np.ndarray) -> float:
    configured_value = EXP1_TARGET_Q_NORM_BY_METHOD[method]
    if configured_value is None:
        return float(np.median(q_norms))
    return float(configured_value)


def _nearest_indices_to_target(q_norms: np.ndarray, target_q_norm: float, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    distances = np.abs(q_norms - target_q_norm)
    candidate_count = min(count, q_norms.shape[0])
    if candidate_count >= q_norms.shape[0]:
        return np.argsort(distances, kind="stable")
    partition = np.argpartition(distances, candidate_count - 1)[:candidate_count]
    return partition[np.argsort(distances[partition], kind="stable")]


def _select_diverse_input_subset(
    x_values: np.ndarray,
    input_norms: np.ndarray,
    candidate_indices: np.ndarray,
    target_count: int,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if target_count >= candidate_indices.size:
        return candidate_indices
    if target_count <= 1:
        return candidate_indices[:1]

    candidate_x = x_values[candidate_indices]
    candidate_input_norms = input_norms[candidate_indices]

    start_local_idx = int(np.argmin(candidate_input_norms))
    selected_local_indices = [start_local_idx]

    if target_count > 1:
        distances_to_start = np.linalg.norm(candidate_x - candidate_x[start_local_idx], axis=1)
        second_local_idx = int(np.argmax(distances_to_start))
        if second_local_idx != start_local_idx:
            selected_local_indices.append(second_local_idx)

    min_distances = np.full(candidate_indices.shape[0], np.inf, dtype=np.float64)
    for local_idx in selected_local_indices:
        distances = np.linalg.norm(candidate_x - candidate_x[local_idx], axis=1)
        min_distances = np.minimum(min_distances, distances)

    selected_local_set = set(selected_local_indices)
    while len(selected_local_indices) < target_count:
        remaining_mask = np.ones(candidate_indices.shape[0], dtype=bool)
        remaining_mask[list(selected_local_set)] = False
        if not bool(np.any(remaining_mask)):
            break

        candidate_scores = min_distances.copy()
        candidate_scores[~remaining_mask] = -np.inf
        next_local_idx = int(np.argmax(candidate_scores))
        if not np.isfinite(candidate_scores[next_local_idx]):
            break

        selected_local_indices.append(next_local_idx)
        selected_local_set.add(next_local_idx)
        distances = np.linalg.norm(candidate_x - candidate_x[next_local_idx], axis=1)
        min_distances = np.minimum(min_distances, distances)

    selected_indices = candidate_indices[np.asarray(selected_local_indices, dtype=np.int64)]
    return selected_indices[np.argsort(input_norms[selected_indices], kind="stable")]


def _raw_target_selection_record(
    setting_key: str,
    method: str,
    input_norms: np.ndarray,
    q_norms: np.ndarray,
) -> dict[str, float | int | str | np.ndarray]:
    target_q_norm = _resolve_target_q_norm(method, q_norms)
    base_tol = _target_mode_base_tol(q_norms)

    selected_indices: np.ndarray | None = None
    used_tol = base_tol
    expansion_idx = 0
    for expansion_idx in range(EXP1_TARGET_Q_NORM_MAX_EXPANSIONS):
        used_tol = base_tol * (2**expansion_idx)
        candidate_indices = np.flatnonzero(np.abs(q_norms - target_q_norm) <= used_tol)
        if candidate_indices.size > 0:
            selected_indices = candidate_indices
            break

    if selected_indices is None or selected_indices.size == 0:
        selected_indices = _nearest_indices_to_target(q_norms, target_q_norm, count=1)

    selected_inputs = input_norms[selected_indices]
    selected_q_norms = q_norms[selected_indices]
    input_span_total = max(float(np.max(input_norms) - np.min(input_norms)), 1e-12)
    input_span = float(np.max(selected_inputs) - np.min(selected_inputs)) if selected_indices.size > 1 else 0.0

    return {
        "setting": setting_key,
        "method": method,
        "bucket_strategy": "target_q_norm_qkmap_cap",
        "selection_subsample_strategy": "all_candidates_then_diverse_input_subset",
        "target_q_norm": target_q_norm,
        "bucket_center": target_q_norm,
        "bucket_width": float(2.0 * used_tol),
        "bucket_lower": float(target_q_norm - used_tol),
        "bucket_upper": float(target_q_norm + used_tol),
        "bucket_count": int(selected_indices.size),
        "bucket_input_min": float(np.min(selected_inputs)),
        "bucket_input_max": float(np.max(selected_inputs)),
        "bucket_input_span": input_span,
        "bucket_input_span_ratio": input_span / input_span_total,
        "bucket_q_norm_mean": float(np.mean(selected_q_norms)),
        "bucket_q_norm_std": float(np.std(selected_q_norms)),
        "bucket_q_norm_std_over_width": float(np.std(selected_q_norms)) / max(2.0 * used_tol, 1e-12),
        "bucket_expansion_step": int(expansion_idx),
        "bucket_score": float(-np.mean(np.abs(selected_q_norms - target_q_norm))),
        "selected_indices": selected_indices,
        "selection_cap_count": int(selected_indices.size),
    }


def _select_target_q_norm_buckets(
    setting_key: str,
    analysis_bundle: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    raw_records = {
        method: _raw_target_selection_record(
            setting_key,
            method,
            analysis_bundle[method]["input_norm"],
            analysis_bundle[method]["q_norm"],
        )
        for method in METHODS
    }

    cap_count = int(raw_records[EXP1_TARGET_Q_NORM_CAP_METHOD]["bucket_count"])
    cap_count = max(cap_count, 1)

    finalized_records: list[dict[str, float | int | str | np.ndarray]] = []
    for method in METHODS:
        record = dict(raw_records[method])
        q_norms = analysis_bundle[method]["q_norm"]
        selected_indices = np.asarray(record["selected_indices"], dtype=np.int64)
        if selected_indices.size > cap_count:
            selected_indices = _select_diverse_input_subset(
                analysis_bundle[method]["x"],
                analysis_bundle[method]["input_norm"],
                selected_indices,
                target_count=cap_count,
            )

        selected_inputs = analysis_bundle[method]["input_norm"][selected_indices]
        selected_q_norms = q_norms[selected_indices]
        input_span_total = max(
            float(np.max(analysis_bundle[method]["input_norm"]) - np.min(analysis_bundle[method]["input_norm"])),
            1e-12,
        )
        input_span = float(np.max(selected_inputs) - np.min(selected_inputs)) if selected_indices.size > 1 else 0.0

        record["selected_indices"] = selected_indices
        record["bucket_count"] = int(selected_indices.size)
        record["selection_cap_count"] = int(cap_count)
        record["bucket_input_min"] = float(np.min(selected_inputs))
        record["bucket_input_max"] = float(np.max(selected_inputs))
        record["bucket_input_span"] = input_span
        record["bucket_input_span_ratio"] = input_span / input_span_total
        record["bucket_q_norm_mean"] = float(np.mean(selected_q_norms))
        record["bucket_q_norm_std"] = float(np.std(selected_q_norms))
        record["bucket_q_norm_std_over_width"] = float(np.std(selected_q_norms)) / max(float(record["bucket_width"]), 1e-12)
        finalized_records.append(record)

    return pd.DataFrame(finalized_records)


def _select_buckets_for_setting(
    setting_key: str,
    analysis_bundle: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    if EXP1_BUCKET_MODE == "auto_max_span_same_norm":
        return pd.DataFrame(
            [
                _select_norm_bucket(
                    setting_key,
                    method,
                    analysis_bundle[method]["input_norm"],
                    analysis_bundle[method]["q_norm"],
                )
                for method in METHODS
            ]
        )
    if EXP1_BUCKET_MODE == "target_q_norm_qkmap_cap":
        return _select_target_q_norm_buckets(setting_key, analysis_bundle)
    raise ValueError(f"Unsupported EXP1_BUCKET_MODE: {EXP1_BUCKET_MODE}")


def _evaluate_setting(
    setting_key: str,
    setting_title: str,
    samples_by_method: dict[str, torch.Tensor],
    operators: dict[str, torch.nn.Module],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    record_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, float | str | int]] = []
    analysis_bundle: dict[str, dict[str, np.ndarray]] = {}

    for method, operator in operators.items():
        samples = samples_by_method[method]
        total_samples = samples.shape[0]
        batch_size = exp1_effective_batch_size(method)
        total_batches = total_samples // batch_size
        input_norms = torch.linalg.vector_norm(samples, dim=-1).cpu()
        batch_q_outputs: list[torch.Tensor] = []
        for batch_idx, batch_start in enumerate(range(0, total_samples, batch_size)):
            batch_end = batch_start + batch_size
            if _should_report(batch_idx, total_batches):
                print(
                    f"[exp1] {setting_key} | {method} | batch {batch_idx + 1}/{total_batches}",
                    flush=True,
                )

            q = operator(samples[batch_start:batch_end], output="q").detach().cpu()
            batch_q_outputs.append(q)

        q_values = torch.cat(batch_q_outputs, dim=0)
        q_norms = torch.linalg.vector_norm(q_values, dim=-1)
        record_frames.append(
            pd.DataFrame(
                {
                    "setting": setting_key,
                    "method": method,
                    "sample_index": list(range(total_samples)),
                    "input_norm": input_norms.tolist(),
                    "q_norm": q_norms.tolist(),
                }
            )
        )
        summary_records.append(
            {
                "setting": setting_key,
                "title": setting_title,
                "method": method,
                "num_samples": int(total_samples),
                "input_norm_min": float(input_norms.min().item()),
                "input_norm_max": float(input_norms.max().item()),
                "q_norm_min": float(q_norms.min().item()),
                "q_norm_max": float(q_norms.max().item()),
                "q_norm_mean": float(q_norms.mean().item()),
                "q_norm_std": float(q_norms.std(unbiased=False).item()),
            }
        )
        analysis_bundle[method] = {
            "x": samples.detach().cpu().numpy(),
            "input_norm": input_norms.numpy(),
            "q_norm": q_norms.numpy(),
            "q": q_values.numpy(),
        }

    return pd.concat(record_frames, ignore_index=True), pd.DataFrame(summary_records), analysis_bundle


def run_experiment_1() -> tuple[pd.DataFrame, dict[str, Path]]:
    validate_config()
    ensure_output_dirs()

    operators = build_all_operators(EXP1_OPERATOR_SEED)
    sample_counts = {method: exp1_num_samples(method) for method in METHODS}
    batch_sizes = {method: exp1_effective_batch_size(method) for method in METHODS}
    print(
        f"[exp1] start: sample_counts={sample_counts}, batch_sizes={batch_sizes}, device={RUNTIME_DEVICE}",
        flush=True,
    )

    setting_specs = [
        (
            "same_direction",
            "Experiment 1: Shared Input Direction",
            "same_direction",
            "exp1_scatter_same_direction.pdf",
            0,
        ),
        (
            "independent_direction",
            "Experiment 1: Independently Sampled Directions",
            "independent_direction",
            "exp1_scatter_independent_direction.pdf",
            1,
        ),
    ]

    all_summaries: list[pd.DataFrame] = []
    output_paths: dict[str, Path] = {}

    with torch.no_grad():
        for setting_key, title, builder_kind, output_name, generator_offset in setting_specs:
            generator = exp1_data_generator(generator_offset)
            scale_tensors = {method: exp1_scale_grid_tensor(method) for method in METHODS}
            max_num_samples = max(scale_tensor.shape[0] for scale_tensor in scale_tensors.values())

            if builder_kind == "same_direction":
                shared_direction = sample_unit_direction(generator)
                samples_by_method = {
                    method: build_scaled_samples(scale_tensors[method], shared_direction)
                    for method in METHODS
                }
            else:
                shared_directions = sample_unit_directions(max_num_samples, generator)
                samples_by_method = {
                    method: build_scaled_samples(
                        scale_tensors[method],
                        shared_directions[: scale_tensors[method].shape[0]],
                    )
                    for method in METHODS
                }

            records_df, summary_df, analysis_bundle = _evaluate_setting(
                setting_key,
                title,
                samples_by_method,
                operators,
            )
            bucket_df = _select_buckets_for_setting(setting_key, analysis_bundle)
            combined_summary_df = summary_df.merge(
                bucket_df,
                on=["setting", "method"],
                how="left",
                validate="one_to_one",
            )
            all_summaries.append(combined_summary_df)

            selection_title = (
                f"{title}: Selected Same-Norm Buckets"
                if EXP1_BUCKET_MODE == "auto_max_span_same_norm"
                else f"{title}: Selected Samples Near Target Query Norm"
            )
            analysis_suffix = (
                "Inside Selected Buckets"
                if EXP1_BUCKET_MODE == "auto_max_span_same_norm"
                else "For Selected Samples Near Target Query Norm"
            )

            output_paths[f"{setting_key}.scatter"] = plot_exp1_scatter_from_records(
                records_df,
                title=title,
                output_name=output_name,
            )
            output_paths[f"{setting_key}.bucket_selection"] = plot_exp1_bucket_selection(
                records_df,
                bucket_df,
                title=selection_title,
                output_name=f"exp1_bucket_selection_{setting_key}.pdf",
            )
            output_paths[f"{setting_key}.bucket_heatmap"] = plot_exp1_bucket_heatmap(
                analysis_bundle,
                bucket_df,
                title=f"{title}: Query Heatmaps {analysis_suffix}",
                output_name=f"exp1_bucket_heatmap_{setting_key}.pdf",
            )
            output_paths[f"{setting_key}.bucket_dim_stats"] = plot_exp1_bucket_dim_stats(
                analysis_bundle,
                bucket_df,
                title=f"{title}: Top Varying Query Dimensions {analysis_suffix}",
                output_name=f"exp1_bucket_dim_stats_{setting_key}.pdf",
            )
            output_paths[f"{setting_key}.bucket_dim_violin"] = plot_exp1_bucket_dim_violin(
                analysis_bundle,
                bucket_df,
                title=f"{title}: Violin Plots of Top Varying Query Dimensions {analysis_suffix}",
                output_name=f"exp1_bucket_dim_violin_{setting_key}.pdf",
            )
            output_paths[f"{setting_key}.bucket_dim_mean_std"] = plot_exp1_bucket_dim_mean_std(
                analysis_bundle,
                bucket_df,
                title=f"{title}: Mean ± Std of Top Varying Query Dimensions {analysis_suffix}",
                output_name=f"exp1_bucket_dim_mean_std_{setting_key}.pdf",
            )
            output_paths[f"{setting_key}.bucket_pairwise"] = plot_exp1_bucket_pairwise_distances(
                analysis_bundle,
                bucket_df,
                title=f"{title}: Pairwise Distances {analysis_suffix}",
                output_name=f"exp1_bucket_pairwise_{setting_key}.pdf",
                sample_count=EXP1_BUCKET_PAIRWISE_SAMPLE_COUNT,
            )
            output_paths[f"{setting_key}.bucket_pairwise_kde"] = plot_exp1_bucket_pairwise_kde(
                analysis_bundle,
                bucket_df,
                title=f"{title}: Pairwise Distance KDE {analysis_suffix}",
                output_name=f"exp1_bucket_pairwise_kde_{setting_key}.pdf",
                sample_count=EXP1_BUCKET_PAIRWISE_SAMPLE_COUNT,
            )
            output_paths[f"{setting_key}.bucket_pca"] = plot_exp1_bucket_pca(
                analysis_bundle,
                bucket_df,
                title=f"{title}: PCA Projections {analysis_suffix}",
                output_name=f"exp1_bucket_pca_{setting_key}.pdf",
                sample_count=EXP1_BUCKET_PCA_SAMPLE_COUNT,
            )
            output_paths[f"{setting_key}.bucket_tsne"] = plot_exp1_bucket_tsne(
                analysis_bundle,
                bucket_df,
                title=f"{title}: t-SNE Projections {analysis_suffix}",
                output_name=f"exp1_bucket_tsne_{setting_key}.pdf",
                sample_count=EXP1_BUCKET_TSNE_SAMPLE_COUNT,
            )
            print(f"[exp1] saved figures for {setting_key}", flush=True)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    print("[exp1] done", flush=True)
    return summary_df, output_paths


def main() -> None:
    summary_df, output_paths = run_experiment_1()
    print("exp1_summary (in memory only)")
    print(summary_df.to_string(index=False))
    print()
    print("exp1_figures")
    for setting_key, output_path in sorted(output_paths.items()):
        print(f"{setting_key}: {output_path}")


if __name__ == "__main__":
    main()
