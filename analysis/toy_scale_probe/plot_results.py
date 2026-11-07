from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from config import (
    EXP1_BUCKET_KDE_GRID_SIZE,
    EXP1_BUCKET_TOPK_DIMS,
    EXP1_BUCKET_VIS_SAMPLE_COUNT,
    EXP1_SCALE_MIN,
    EXP1_QKRN_DENSE_MAX,
    EXP1_TSNE_LEARNING_RATE,
    EXP1_TSNE_MAX_ITER,
    EXP1_TSNE_PERPLEXITY,
    FIG_PATH,
    METHOD_DISPLAY_NAMES,
    METHODS,
    OUTPUT_PATH,
    ensure_output_dirs,
)


def _method_labels() -> list[str]:
    return [METHOD_DISPLAY_NAMES[method] for method in METHODS]


def _boxplot(ax: Axes, data: list[np.ndarray], ylabel: str, title: str, ylim=None) -> None:
    ax.boxplot(
        data,
        whis=(0, 100),
        tick_labels=_method_labels(),
        showfliers=False,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


# def _exp1_method_ylim(method: str, q_norms: np.ndarray) -> tuple[float, float]:
#     max_value = float(np.max(q_norms))
#     if method == "cosine":
#         return 0.0, max(1.05, 1.02 * max_value)
#     if max_value <= 0.0:
#         return -0.05, 0.05
#     return 0.0, 1.05 * max_value

def _exp1_method_ylim(method: str, q_norms: np.ndarray) -> tuple[float, float]:
    min_value = float(np.min(q_norms))
    max_value = float(np.max(q_norms))
    span = max_value - min_value

    if span <= 1e-12:
        center = 0.5 * (min_value + max_value)
        padding = max(abs(center) * 0.05, 1e-6)
    else:
        padding = 0.05 * span

    return min_value - padding, max_value + padding

def _qkrn_zoom_bounds(input_norms: np.ndarray, q_norms: np.ndarray) -> tuple[float, float, float, float] | None:
    if input_norms.size < 4:
        return None

    order = np.argsort(input_norms)
    x_sorted = input_norms[order]
    y_sorted = q_norms[order]
    zoom_mask = (x_sorted >= EXP1_SCALE_MIN) & (x_sorted <= EXP1_QKRN_DENSE_MAX)
    if int(np.count_nonzero(zoom_mask)) < 4:
        return None

    zoom_y = y_sorted[zoom_mask]
    zoom_y_max = float(np.max(zoom_y))
    zoom_span = max(zoom_y_max, 1e-6)
    padding = 0.05 * zoom_span
    return EXP1_SCALE_MIN, EXP1_QKRN_DENSE_MAX, 0.0, zoom_y_max + padding


def _add_qkrn_zoom_inset(ax: Axes, method_df: pd.DataFrame) -> None:
    zoom_bounds = _qkrn_zoom_bounds(
        method_df["input_norm"].to_numpy(),
        method_df["q_norm"].to_numpy(),
    )
    if zoom_bounds is None:
        return

    x0, x1, y0, y1 = zoom_bounds
    inset = inset_axes(ax, width="42%", height="42%", loc="center right", borderpad=1.1)
    inset.scatter(method_df["input_norm"], method_df["q_norm"], s=12, alpha=0.8, edgecolors="none")
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.grid(True, linestyle="--", alpha=0.25)
    inset.tick_params(labelsize=7)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.8)


def plot_exp1_scatter_from_records(
    records_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    x_min = float(records_df["input_norm"].min())
    x_max = float(records_df["input_norm"].max())

    for ax, method in zip(axes.flat, METHODS):
        method_df = records_df.loc[records_df["method"] == method].sort_values("input_norm")
        q_norms = method_df["q_norm"].to_numpy()
        ax.scatter(method_df["input_norm"], q_norms, s=18, alpha=0.82, edgecolors="none")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*_exp1_method_ylim(method, q_norms))
        ax.set_title(METHOD_DISPLAY_NAMES[method])
        ax.set_xlabel("Input norm ||x||")
        ax.set_ylabel("Query norm ||q||")
        ax.grid(True, linestyle="--", alpha=0.3)

        if method == "cosine":
            ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0, alpha=0.55)
        if method == "qkrn":
            _add_qkrn_zoom_inset(ax, method_df)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92, wspace=0.24, hspace=0.28)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _bucket_row(bucket_df: pd.DataFrame, method: str) -> pd.Series:
    method_rows = bucket_df.loc[bucket_df["method"] == method]
    if method_rows.empty:
        raise KeyError(f"Missing bucket specification for method={method}")
    return method_rows.iloc[0]


def _selected_indices(bucket_row: pd.Series, bundle: dict[str, np.ndarray]) -> np.ndarray:
    raw_indices = bucket_row.get("selected_indices")
    if raw_indices is None:
        q_norm = bundle["q_norm"]
        lower = float(bucket_row["bucket_lower"])
        upper = float(bucket_row["bucket_upper"])
        return np.flatnonzero((q_norm >= lower) & (q_norm <= upper))
    return np.asarray(raw_indices, dtype=np.int64)


def _bucket_mask(bundle: dict[str, np.ndarray], bucket_row: pd.Series) -> np.ndarray:
    mask = np.zeros(bundle["q_norm"].shape[0], dtype=bool)
    selected = _selected_indices(bucket_row, bundle)
    mask[selected] = True
    return mask


def _evenly_spaced_indices(num_items: int, target_count: int) -> np.ndarray:
    count = min(num_items, target_count)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == num_items:
        return np.arange(num_items, dtype=np.int64)
    return np.linspace(0, num_items - 1, count, dtype=np.int64)


def _representative_bucket_vectors(
    bundle: dict[str, np.ndarray],
    bucket_row: pd.Series,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = _selected_indices(bucket_row, bundle)
    q = bundle["q"][selected]
    input_norm = bundle["input_norm"][selected]
    q_norm = bundle["q_norm"][selected]

    if q.shape[0] == 0:
        raise ValueError("Selected bucket contains no samples")

    order = np.argsort(input_norm)
    if bucket_row.get("bucket_strategy") == "target_q_norm_qkmap_cap":
        selected_order = np.arange(order.size, dtype=np.int64)
    else:
        selected_order = _evenly_spaced_indices(order.size, max_samples)
    chosen = order[selected_order]
    return q[chosen], input_norm[chosen], q_norm[chosen]


def _bucket_title_line(bucket_row: pd.Series) -> str:
    if bucket_row.get("bucket_strategy") == "target_q_norm_qkmap_cap":
        return (
            f"{METHOD_DISPLAY_NAMES[bucket_row['method']]} | "
            f"target={bucket_row['target_q_norm']:.6g} | "
            f"band=[{bucket_row['bucket_lower']:.6g}, {bucket_row['bucket_upper']:.6g}] | "
            f"selected={int(bucket_row['bucket_count'])}"
        )
    return (
        f"{METHOD_DISPLAY_NAMES[bucket_row['method']]} | "
        f"bucket=[{bucket_row['bucket_lower']:.6g}, {bucket_row['bucket_upper']:.6g}] | "
        f"n={int(bucket_row['bucket_count'])}"
    )


def _annotate_too_few_samples(ax: Axes, method: str, count: int, required: int) -> None:
    ax.set_title(f"{METHOD_DISPLAY_NAMES[method]} | selected={count}", fontsize=9)
    ax.text(
        0.5,
        0.5,
        f"Too few samples\nneed >= {required}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def plot_exp1_bucket_selection(
    records_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    x_min = float(records_df["input_norm"].min())
    x_max = float(records_df["input_norm"].max())

    for ax, method in zip(axes.flat, METHODS):
        method_df = records_df.loc[records_df["method"] == method].sort_values("input_norm")
        bucket_row = _bucket_row(bucket_df, method)
        selected_index_set = set(int(index) for index in np.asarray(bucket_row["selected_indices"], dtype=np.int64))
        selected_mask = method_df["sample_index"].isin(selected_index_set).to_numpy()
        ax.scatter(
            method_df["input_norm"],
            method_df["q_norm"],
            s=12,
            alpha=0.18,
            color="tab:blue",
            edgecolors="none",
        )
        ax.scatter(
            method_df.loc[selected_mask, "input_norm"],
            method_df.loc[selected_mask, "q_norm"],
            s=18,
            alpha=0.85,
            color="tab:red",
            edgecolors="none",
        )
        ax.axhspan(
            float(bucket_row["bucket_lower"]),
            float(bucket_row["bucket_upper"]),
            color="tab:red",
            alpha=0.10,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*_exp1_method_ylim(method, method_df["q_norm"].to_numpy()))
        ax.set_title(_bucket_title_line(bucket_row), fontsize=9)
        ax.set_xlabel("Input norm ||x||")
        ax.set_ylabel("Query norm ||q||")
        ax.grid(True, linestyle="--", alpha=0.3)

        if method == "cosine":
            ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0, alpha=0.5)
        if method == "qkrn":
            _add_qkrn_zoom_inset(ax, method_df)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92, wspace=0.24, hspace=0.30)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp1_bucket_heatmap(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0))
    images = []
    for ax, method in zip(axes.flat, METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, input_norms, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            EXP1_BUCKET_VIS_SAMPLE_COUNT,
        )
        color_limit = np.percentile(np.abs(vectors), 99)
        color_limit = float(max(color_limit, 1e-6))
        image = ax.imshow(
            vectors,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        images.append(image)
        ax.set_title(_bucket_title_line(bucket_row), fontsize=9)
        ax.set_xlabel("Query dimension")
        ax.set_ylabel("Bucket sample (sorted by ||x||)")

        row_ticks = np.linspace(0, len(input_norms) - 1, min(5, len(input_norms)), dtype=int)
        ax.set_yticks(row_ticks)
        ax.set_yticklabels([f"{input_norms[idx]:.2f}" for idx in row_ticks])

    for ax, image in zip(axes.flat, images):
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.92, wspace=0.22, hspace=0.32)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp1_bucket_dim_stats(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
    for ax, method in zip(axes.flat, METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, _, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            EXP1_BUCKET_VIS_SAMPLE_COUNT,
        )
        dim_std = np.std(vectors, axis=0)
        topk = min(EXP1_BUCKET_TOPK_DIMS, vectors.shape[1])
        top_indices = np.argsort(dim_std)[-topk:]
        top_indices = top_indices[np.argsort(dim_std[top_indices])[::-1]]
        top_values = [vectors[:, dim_idx] for dim_idx in top_indices]

        ax.boxplot(
            top_values,
            whis=(0, 100),
            showfliers=False,
            widths=0.6,
            medianprops={"color": "black", "linewidth": 1.3},
        )
        ax.set_title(_bucket_title_line(bucket_row), fontsize=9)
        ax.set_xlabel("Top varying dimensions")
        ax.set_ylabel("Component value")
        ax.set_xticks(range(1, topk + 1))
        ax.set_xticklabels([str(int(dim_idx)) for dim_idx in top_indices], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.92, wspace=0.24, hspace=0.30)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp1_bucket_dim_violin(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
    for ax, method in zip(axes.flat, METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, _, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            EXP1_BUCKET_VIS_SAMPLE_COUNT,
        )
        dim_std = np.std(vectors, axis=0)
        topk = min(EXP1_BUCKET_TOPK_DIMS, vectors.shape[1])
        top_indices = np.argsort(dim_std)[-topk:]
        top_indices = top_indices[np.argsort(dim_std[top_indices])[::-1]]
        top_values = [vectors[:, dim_idx] for dim_idx in top_indices]

        violin = ax.violinplot(top_values, showmeans=False, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("tab:blue")
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        violin["cmedians"].set_color("black")
        violin["cmedians"].set_linewidth(1.2)

        ax.set_title(_bucket_title_line(bucket_row), fontsize=9)
        ax.set_xlabel("Top varying dimensions")
        ax.set_ylabel("Component value")
        ax.set_xticks(range(1, topk + 1))
        ax.set_xticklabels([str(int(dim_idx)) for dim_idx in top_indices], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.92, wspace=0.24, hspace=0.30)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp1_bucket_dim_mean_std(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))
    for ax, method in zip(axes.flat, METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, _, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            EXP1_BUCKET_VIS_SAMPLE_COUNT,
        )
        dim_std = np.std(vectors, axis=0)
        topk = min(EXP1_BUCKET_TOPK_DIMS, vectors.shape[1])
        top_indices = np.argsort(dim_std)[-topk:]
        top_indices = top_indices[np.argsort(dim_std[top_indices])[::-1]]

        means = np.mean(vectors[:, top_indices], axis=0)
        stds = np.std(vectors[:, top_indices], axis=0)
        x = np.arange(topk)

        ax.plot(x, means, color="tab:blue", linewidth=1.8, marker="o", markersize=4)
        ax.fill_between(x, means - stds, means + stds, color="tab:blue", alpha=0.20)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title(_bucket_title_line(bucket_row), fontsize=9)
        ax.set_xlabel("Top varying dimensions")
        ax.set_ylabel("Mean ± std")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(dim_idx)) for dim_idx in top_indices], rotation=45, ha="right")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.92, wspace=0.24, hspace=0.30)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _pairwise_distance_summary(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = vectors @ vectors.T
    sq_norm = np.sum(vectors * vectors, axis=1, keepdims=True)
    sq_dist = np.maximum(sq_norm + sq_norm.T - 2.0 * gram, 0.0)
    euclidean = np.sqrt(sq_dist)

    norms = np.sqrt(np.maximum(sq_norm, 1e-12))
    cosine_similarity = gram / np.maximum(norms * norms.T, 1e-12)
    cosine_distance = 1.0 - np.clip(cosine_similarity, -1.0, 1.0)

    upper = np.triu_indices(vectors.shape[0], k=1)
    return euclidean[upper], cosine_distance[upper]


def _gaussian_kde_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.linspace(0.0, 1.0, EXP1_BUCKET_KDE_GRID_SIZE), np.zeros(EXP1_BUCKET_KDE_GRID_SIZE)

    v_min = float(np.min(values))
    v_max = float(np.max(values))
    v_std = float(np.std(values))
    if not np.isfinite(v_std) or v_std < 1e-12:
        v_std = 1e-3
    if abs(v_max - v_min) < 1e-12:
        padding = 3.0 * v_std
        grid = np.linspace(v_min - padding, v_max + padding, EXP1_BUCKET_KDE_GRID_SIZE)
    else:
        padding = 0.05 * (v_max - v_min)
        grid = np.linspace(v_min - padding, v_max + padding, EXP1_BUCKET_KDE_GRID_SIZE)

    bandwidth = 1.06 * v_std * max(values.size, 2) ** (-1.0 / 5.0)
    bandwidth = float(max(bandwidth, 1e-3))
    normalized = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * normalized * normalized).mean(axis=1) / (bandwidth * np.sqrt(2.0 * np.pi))
    return grid, density


def plot_exp1_bucket_pairwise_distances(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    sample_count: int,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(len(METHODS), 2, figsize=(11.8, 11.8))
    for row_idx, method in enumerate(METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, _, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            sample_count,
        )
        if vectors.shape[0] < 2:
            _annotate_too_few_samples(axes[row_idx, 0], method, vectors.shape[0], required=2)
            _annotate_too_few_samples(axes[row_idx, 1], method, vectors.shape[0], required=2)
            continue
        euclidean, cosine_distance = _pairwise_distance_summary(vectors)

        euclidean_ax = axes[row_idx, 0]
        cosine_ax = axes[row_idx, 1]

        euclidean_ax.hist(euclidean, bins=30, color="tab:blue", alpha=0.8)
        euclidean_ax.set_title(
            f"{METHOD_DISPLAY_NAMES[method]} | Euclidean\nmean={euclidean.mean():.4g}, std={euclidean.std():.4g}",
            fontsize=9,
        )
        euclidean_ax.set_xlabel("Pairwise Euclidean distance")
        euclidean_ax.set_ylabel("Count")
        euclidean_ax.grid(axis="y", linestyle="--", alpha=0.25)

        cosine_ax.hist(cosine_distance, bins=30, color="tab:orange", alpha=0.8)
        cosine_ax.set_title(
            f"{METHOD_DISPLAY_NAMES[method]} | Cosine distance\nmean={cosine_distance.mean():.4g}, std={cosine_distance.std():.4g}",
            fontsize=9,
        )
        cosine_ax.set_xlabel("Pairwise cosine distance")
        cosine_ax.set_ylabel("Count")
        cosine_ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.94, wspace=0.24, hspace=0.45)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp1_bucket_pairwise_kde(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    sample_count: int,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(len(METHODS), 2, figsize=(11.8, 11.8))
    for row_idx, method in enumerate(METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, _, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            sample_count,
        )
        if vectors.shape[0] < 2:
            _annotate_too_few_samples(axes[row_idx, 0], method, vectors.shape[0], required=2)
            _annotate_too_few_samples(axes[row_idx, 1], method, vectors.shape[0], required=2)
            continue
        euclidean, cosine_distance = _pairwise_distance_summary(vectors)
        euclidean_grid, euclidean_density = _gaussian_kde_curve(euclidean)
        cosine_grid, cosine_density = _gaussian_kde_curve(cosine_distance)

        euclidean_ax = axes[row_idx, 0]
        cosine_ax = axes[row_idx, 1]

        euclidean_ax.plot(euclidean_grid, euclidean_density, color="tab:blue", linewidth=1.8)
        euclidean_ax.fill_between(euclidean_grid, 0.0, euclidean_density, color="tab:blue", alpha=0.18)
        euclidean_ax.set_title(
            f"{METHOD_DISPLAY_NAMES[method]} | Euclidean KDE\nmean={euclidean.mean():.4g}, std={euclidean.std():.4g}",
            fontsize=9,
        )
        euclidean_ax.set_xlabel("Pairwise Euclidean distance")
        euclidean_ax.set_ylabel("Density")
        euclidean_ax.grid(True, linestyle="--", alpha=0.25)

        cosine_ax.plot(cosine_grid, cosine_density, color="tab:orange", linewidth=1.8)
        cosine_ax.fill_between(cosine_grid, 0.0, cosine_density, color="tab:orange", alpha=0.18)
        cosine_ax.set_title(
            f"{METHOD_DISPLAY_NAMES[method]} | Cosine KDE\nmean={cosine_distance.mean():.4g}, std={cosine_distance.std():.4g}",
            fontsize=9,
        )
        cosine_ax.set_xlabel("Pairwise cosine distance")
        cosine_ax.set_ylabel("Density")
        cosine_ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.94, wspace=0.24, hspace=0.45)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _pca_project(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    projected = centered @ components.T
    explained = (singular_values[:2] ** 2) / max(np.sum(singular_values**2), 1e-12)
    return projected, explained


def plot_exp1_bucket_pca(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    sample_count: int,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    for ax, method in zip(axes.flat, METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, input_norms, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            sample_count,
        )
        if vectors.shape[0] < 2:
            _annotate_too_few_samples(ax, method, vectors.shape[0], required=2)
            continue
        projected, explained = _pca_project(vectors)
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=input_norms,
            cmap="viridis",
            s=22,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(
            f"{METHOD_DISPLAY_NAMES[method]} | PCA var={100 * explained.sum():.2f}%",
            fontsize=9,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Input norm ||x||")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.92, wspace=0.25, hspace=0.30)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _tsne_project(vectors: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE

    num_samples = vectors.shape[0]
    perplexity = min(EXP1_TSNE_PERPLEXITY, max(5.0, float(num_samples - 1) / 3.0))
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=EXP1_TSNE_LEARNING_RATE,
        max_iter=EXP1_TSNE_MAX_ITER,
        init="pca",
        random_state=0,
    )
    return model.fit_transform(vectors)


def plot_exp1_bucket_tsne(
    analysis_bundle: dict[str, dict[str, np.ndarray]],
    bucket_df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    sample_count: int,
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    for ax, method in zip(axes.flat, METHODS):
        bucket_row = _bucket_row(bucket_df, method)
        vectors, input_norms, _ = _representative_bucket_vectors(
            analysis_bundle[method],
            bucket_row,
            sample_count,
        )
        if vectors.shape[0] < 3:
            _annotate_too_few_samples(ax, method, vectors.shape[0], required=3)
            continue
        projected = _tsne_project(vectors)
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=input_norms,
            cmap="plasma",
            s=22,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(
            f"{METHOD_DISPLAY_NAMES[method]} | t-SNE",
            fontsize=9,
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, linestyle="--", alpha=0.25)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Input norm ||x||")

    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.92, wspace=0.25, hspace=0.30)
    output_path = FIG_PATH / output_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp2_logit_vs_gamma(curves_csv: Path | None = None) -> Path:
    ensure_output_dirs()
    curves_path = curves_csv or OUTPUT_PATH / "exp2_curves.csv"
    df = pd.read_csv(curves_path)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), sharex=True, sharey=False)
    cmap = plt.get_cmap("tab10")

    any_non_positive = bool((df["S_01"] <= 0).any())
    for color_idx, (ax, method) in enumerate(zip(axes, METHODS)):
        method_df = df.loc[df["method"] == method].copy()
        grouped = (
            method_df.groupby("gamma")["S_01"]
            .agg(
                min_value="min",
                q25=lambda values: np.quantile(values, 0.25),
                median="median",
                q75=lambda values: np.quantile(values, 0.75),
                max_value="max",
            )
            .reset_index()
            .sort_values("gamma")
        )
        color = cmap(color_idx)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        # if any_non_positive:
        #     ax.set_yscale("symlog", linthresh=1e-3)
        # else:
        #     ax.set_yscale("log")
        gammas = grouped["gamma"].to_numpy()
        min_values = grouped["min_value"].to_numpy()
        q25_values = grouped["q25"].to_numpy()
        median_values = grouped["median"].to_numpy()
        q75_values = grouped["q75"].to_numpy()
        max_values = grouped["max_value"].to_numpy()

        ax.fill_between(
            gammas,
            min_values,
            q25_values,
            color=color,
            alpha=0.10,
            label="0-25% over seeds",
        )
        ax.fill_between(
            gammas,
            q25_values,
            q75_values,
            color=color,
            alpha=0.25,
            label="IQR (25-75%) over seeds",
        )
        ax.fill_between(
            gammas,
            q75_values,
            max_values,
            color=color,
            alpha=0.10,
            label="75-100% over seeds",
        )
        ax.plot(
            gammas,
            min_values,
            linestyle=":",
            color=color,
            linewidth=1.0,
            label="0% / 25% / 50% / 75% / 100% over seeds",
        )
        ax.plot(
            gammas,
            q25_values,
            linestyle="--",
            color=color,
            linewidth=1.2,
        )
        ax.plot(
            gammas,
            median_values,
            linestyle="--",
            color=color,
            linewidth=1.8,
        )
        ax.plot(
            gammas,
            q75_values,
            linestyle="--",
            color=color,
            linewidth=1.2,
        )
        ax.plot(
            gammas,
            max_values,
            linestyle=":",
            color=color,
            linewidth=1.0,
        )
        ax.set_title(METHOD_DISPLAY_NAMES[method])
        ax.set_xlabel("Gamma")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(
            fontsize=7,
            loc="lower right",        # 图例框的「右下角」去对齐 anchor
            bbox_to_anchor=(1.0, 1.02),  # anchor = 子图右上角略上方；1.0 = 与右边线对齐
            borderaxespad=0.0,
            frameon=True,
        )
        ax.relim()
        ax.autoscale_view()
        ax.margins(y=0.05)

    axes[0].set_ylabel("S_01")
    fig.suptitle("Experiment 2: Logit Response to Key Scaling", y=1.03)
    fig.tight_layout()
    output_path = FIG_PATH / "exp2_logit_vs_gamma.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp2_smax_box(per_seed_csv: Path | None = None) -> Path:
    ensure_output_dirs()
    per_seed_path = per_seed_csv or OUTPUT_PATH / "exp2_per_seed.csv"
    df = pd.read_csv(per_seed_path)
    data = [df.loc[df["method"] == method, "S_max"].to_numpy() for method in METHODS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _boxplot(
        ax,
        data,
        ylabel="S_max = max_gamma |S_01|",
        title="Experiment 2: Max Logit Magnitude",
    )
    fig.tight_layout()
    output_path = FIG_PATH / "exp2_S_max_box.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp2_response_spearman_box(per_seed_csv: Path | None = None) -> Path:
    ensure_output_dirs()
    per_seed_path = per_seed_csv or OUTPUT_PATH / "exp2_per_seed.csv"
    df = pd.read_csv(per_seed_path)
    data = [df.loc[df["method"] == method, "resp_spearman"].to_numpy() for method in METHODS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _boxplot(
        ax,
        data,
        ylabel="Spearman(gamma, S_01) in response window",
        title="Experiment 2: Response Spearman",
        ylim=(-1.05, 1.05),
    )
    fig.tight_layout()
    output_path = FIG_PATH / "exp2_response_spearman_box.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp3_logit_vs_gamma(curves_csv: Path | None = None) -> Path:
    ensure_output_dirs()
    curves_path = curves_csv or OUTPUT_PATH / "exp3_curves.csv"
    output_path = FIG_PATH / "exp3_logit_vs_gamma.pdf"
    if not curves_path.exists():
        if output_path.exists():
            return output_path
        raise FileNotFoundError(f"Missing exp3 curve source: {curves_path}")
    df = pd.read_csv(curves_path)

    curve_values = {method: None for method in METHODS}
    for method in METHODS:
        method_df = df.loc[df["method"] == method, ["gamma", "S_01"]].copy()
        grouped = (
            method_df.groupby("gamma")["S_01"]
            .agg(
                min_value="min",
                q25=lambda values: np.quantile(values, 0.25),
                median="median",
                q75=lambda values: np.quantile(values, 0.75),
                max_value="max",
            )
            .reset_index()
            .sort_values("gamma")
        )
        curve_values[method] = grouped

    return plot_exp3_logit_vs_gamma_from_arrays(
        gammas=curve_values[METHODS[0]]["gamma"].to_numpy(),
        grouped_curves={
            method: {
                "min_value": curve_values[method]["min_value"].to_numpy(),
                "q25": curve_values[method]["q25"].to_numpy(),
                "median": curve_values[method]["median"].to_numpy(),
                "q75": curve_values[method]["q75"].to_numpy(),
                "max_value": curve_values[method]["max_value"].to_numpy(),
            }
            for method in METHODS
        },
    )


def plot_exp3_logit_vs_gamma_from_arrays(
    gammas: np.ndarray,
    grouped_curves: dict[str, dict[str, np.ndarray]],
) -> Path:
    ensure_output_dirs()

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8), sharex=True, sharey=False)
    cmap = plt.get_cmap("tab10")

    any_non_positive = any((grouped_curves[method]["max_value"] <= 0).any() or (grouped_curves[method]["min_value"] <= 0).any() for method in METHODS)
    for color_idx, (ax, method) in enumerate(zip(axes, METHODS)):
        color = cmap(color_idx)
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        # if any_non_positive:
        #     ax.set_yscale("symlog", linthresh=1e-3)
        # else:
        #     ax.set_yscale("log")
        min_values = grouped_curves[method]["min_value"]
        q25_values = grouped_curves[method]["q25"]
        median_values = grouped_curves[method]["median"]
        q75_values = grouped_curves[method]["q75"]
        max_values = grouped_curves[method]["max_value"]

        ax.fill_between(
            gammas,
            min_values,
            q25_values,
            color=color,
            alpha=0.10,
            label="0-25% over seed-angle pairs",
        )
        ax.fill_between(
            gammas,
            q25_values,
            q75_values,
            color=color,
            alpha=0.25,
            label="IQR (25-75%) over seed-angle pairs",
        )
        ax.fill_between(
            gammas,
            q75_values,
            max_values,
            color=color,
            alpha=0.10,
            label="75-100% over seed-angle pairs",
        )
        ax.plot(
            gammas,
            min_values,
            linestyle=":",
            color=color,
            linewidth=1.0,
            label="0% / 25% / 50% / 75% / 100% over seed-angle pairs",
        )
        ax.plot(
            gammas,
            q25_values,
            linestyle="--",
            color=color,
            linewidth=1.2,
        )
        ax.plot(
            gammas,
            median_values,
            linestyle="--",
            color=color,
            linewidth=1.8,
        )
        ax.plot(
            gammas,
            q75_values,
            linestyle="--",
            color=color,
            linewidth=1.2,
        )
        ax.plot(
            gammas,
            max_values,
            linestyle=":",
            color=color,
            linewidth=1.0,
        )
        ax.set_title(METHOD_DISPLAY_NAMES[method])
        ax.set_xlabel("Gamma")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(
            fontsize=7,
            loc="lower right",        # 图例框的「右下角」去对齐 anchor
            bbox_to_anchor=(1.0, 1.02),  # anchor = 子图右上角略上方；1.0 = 与右边线对齐
            borderaxespad=0.0,
            frameon=True,
        )
        ax.relim()
        ax.autoscale_view()
        ax.margins(y=0.05)

    axes[0].set_ylabel("S_01")
    fig.suptitle("Experiment 3: Logit Response with Query-Key Angle Sweep", y=1.03)
    fig.tight_layout()
    output_path = FIG_PATH / "exp3_logit_vs_gamma.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp3_smax_box(per_seed_csv: Path | None = None) -> Path:
    ensure_output_dirs()
    per_seed_path = per_seed_csv or OUTPUT_PATH / "exp3_per_seed.csv"
    output_path = FIG_PATH / "exp3_S_max_box.pdf"
    if not per_seed_path.exists():
        if output_path.exists():
            return output_path
        raise FileNotFoundError(f"Missing exp3 metric source: {per_seed_path}")
    df = pd.read_csv(per_seed_path)
    data = [df.loc[df["method"] == method, "S_max"].to_numpy() for method in METHODS]

    return plot_exp3_smax_box_from_arrays({method: values for method, values in zip(METHODS, data)})


def plot_exp3_smax_box_from_arrays(smax_values: dict[str, np.ndarray]) -> Path:
    ensure_output_dirs()
    data = [np.asarray(smax_values[method]) for method in METHODS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _boxplot(
        ax,
        data,
        ylabel="S_max = max_gamma |S_01|",
        title="Experiment 3: Max Logit Magnitude",
    )
    fig.tight_layout()
    output_path = FIG_PATH / "exp3_S_max_box.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_exp3_response_spearman_box(per_seed_csv: Path | None = None) -> Path:
    ensure_output_dirs()
    per_seed_path = per_seed_csv or OUTPUT_PATH / "exp3_per_seed.csv"
    output_path = FIG_PATH / "exp3_response_spearman_box.pdf"
    if not per_seed_path.exists():
        if output_path.exists():
            return output_path
        raise FileNotFoundError(f"Missing exp3 metric source: {per_seed_path}")
    df = pd.read_csv(per_seed_path)
    data = [df.loc[df["method"] == method, "resp_spearman"].to_numpy() for method in METHODS]

    return plot_exp3_response_spearman_box_from_arrays(
        {method: values for method, values in zip(METHODS, data)}
    )


def plot_exp3_response_spearman_box_from_arrays(resp_spearman_values: dict[str, np.ndarray]) -> Path:
    ensure_output_dirs()
    data = [np.asarray(resp_spearman_values[method]) for method in METHODS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _boxplot(
        ax,
        data,
        ylabel="Spearman(gamma, S_01) in response window",
        title="Experiment 3: Response Spearman",
        ylim=(-1.05, 1.05),
    )
    fig.tight_layout()
    output_path = FIG_PATH / "exp3_response_spearman_box.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_all() -> None:
    plot_exp2_logit_vs_gamma()
    plot_exp2_smax_box()
    plot_exp2_response_spearman_box()
    try:
        plot_exp3_logit_vs_gamma()
        plot_exp3_smax_box()
        plot_exp3_response_spearman_box()
    except FileNotFoundError:
        # Experiment 3 can now render directly from in-memory arrays without persisting CSVs.
        # Skip file-based redraw when those temporary files do not exist.
        pass


def main() -> None:
    plot_all()
    print(f"Saved figures to {FIG_PATH}")


if __name__ == "__main__":
    main()
