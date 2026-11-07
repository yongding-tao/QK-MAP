#!/usr/bin/env python3

"""Controlled scale-response analysis using trained 190M checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM


COLORS = {
    "Baseline": "#4D4D4D",
    "QK-RMSNorm": "#0072B2",
    "QK-MAP p=0": "#999999",
    "QK-MAP p=0.125": "#E69F00",
    "QK-MAP p=0.25": "#009E73",
    "QK-MAP p=0.5": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Model label and checkpoint path. Repeat once per model.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the PDF from scale_response_summary.csv.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--bank-size", type=int, default=32)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    )
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
        models[label.strip()] = Path(raw_path).expanduser()
    return models


def rms_unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)


def project(
    attention: torch.nn.Module,
    x: torch.Tensor,
    which: str,
) -> torch.Tensor:
    if which == "q":
        projection = attention.w_q
        norm = attention.q_norm
    else:
        projection = attention.w_k
        norm = attention.k_norm

    # Some historical HF exports mix dtypes between the wrapper, FAN layer,
    # and Q/K projections. Follow the dtype of each actual weight at its
    # boundary instead of trusting model.dtype.
    if attention.fanlayer is not None:
        fan_dtype = attention.fanlayer.input_linear.weight.dtype
        mapped = attention.fanlayer(x.to(fan_dtype))
    else:
        mapped = x
    value = projection(mapped.to(projection.weight.dtype))
    if norm is not None:
        if getattr(norm, "weight", None) is not None:
            value = value.to(norm.weight.dtype)
        value = norm(value)
    n_heads = attention.n_heads
    return value.unflatten(-1, (n_heads, value.shape[-1] // n_heads))


def mean_and_ci(values: torch.Tensor) -> tuple[float, float, float]:
    flat = values.detach().float().flatten()
    mean = flat.mean().item()
    if flat.numel() < 2:
        return mean, mean, mean
    half = 1.96 * flat.std(unbiased=True).item() / math.sqrt(flat.numel())
    return mean, mean - half, mean + half


@torch.inference_mode()
def analyze_model(
    label: str,
    model_path: Path,
    scales: list[float],
    trials: int,
    bank_size: int,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()

    d_model = model.config.olmo_core_model_dict["d_model"]
    generator = torch.Generator(device=device).manual_seed(seed)

    query_dirs = rms_unit(
        torch.randn(trials, d_model, generator=generator, device=device)
    )
    key_dirs = rms_unit(
        torch.randn(trials, bank_size, d_model, generator=generator, device=device)
    )
    query_bank = rms_unit(
        torch.randn(trials, bank_size, d_model, generator=generator, device=device)
    )
    target_dirs = rms_unit(
        torch.randn(trials, d_model, generator=generator, device=device)
    )
    distractor_dirs = rms_unit(
        torch.randn(
            trials, bank_size - 1, d_model, generator=generator, device=device
        )
    )

    rows: list[dict[str, Any]] = []
    for layer_idx, block in enumerate(model.olmo.blocks.values()):
        attention = block.attention
        head_dim = d_model // attention.n_heads

        fixed_keys = project(attention, key_dirs, "k")
        fixed_queries = project(attention, query_bank, "q")
        distractor_keys = project(attention, distractor_dirs, "k")

        for scale in scales:
            scaled_queries = project(attention, query_dirs * scale, "q")
            query_logits = torch.einsum(
                "thd,tkhd->thk", scaled_queries, fixed_keys
            ).float() / math.sqrt(head_dim)
            query_probs = query_logits.softmax(dim=-1)
            normalized_entropy = (
                -(query_probs * query_probs.clamp_min(1e-12).log()).sum(dim=-1)
                / math.log(bank_size)
            )

            target_keys = project(attention, target_dirs * scale, "k")
            all_keys = torch.cat([target_keys[:, None], distractor_keys], dim=1)
            key_logits = torch.einsum(
                "tqhd,tkhd->tqhk", fixed_queries, all_keys
            ).float() / math.sqrt(head_dim)
            target_mass = key_logits.softmax(dim=-1)[..., 0].mean(dim=1)
            target_mass_ratio = target_mass * bank_size

            q_rms = scaled_queries.float().square().mean(dim=-1).sqrt()
            k_rms = target_keys.float().square().mean(dim=-1).sqrt()

            q_ent = mean_and_ci(normalized_entropy)
            k_mass = mean_and_ci(target_mass_ratio)
            q_norm = mean_and_ci(q_rms)
            k_norm = mean_and_ci(k_rms)
            rows.append(
                {
                    "model": label,
                    "layer": layer_idx,
                    "scale": scale,
                    "query_entropy_mean": q_ent[0],
                    "query_entropy_ci_low": q_ent[1],
                    "query_entropy_ci_high": q_ent[2],
                    "target_key_mass_ratio_mean": k_mass[0],
                    "target_key_mass_ratio_ci_low": k_mass[1],
                    "target_key_mass_ratio_ci_high": k_mass[2],
                    "q_rms_mean": q_norm[0],
                    "q_rms_ci_low": q_norm[1],
                    "q_rms_ci_high": q_norm[2],
                    "k_rms_mean": k_norm[0],
                    "k_rms_ci_low": k_norm[1],
                    "k_rms_ci_high": k_norm[2],
                }
            )

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def aggregate(
    rows: list[dict[str, Any]], model_order: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["scale"]), []).append(row)

    result: list[dict[str, Any]] = []
    metrics = [
        "query_entropy_mean",
        "target_key_mass_ratio_mean",
        "q_rms_mean",
        "k_rms_mean",
    ]
    for (model, scale), group in grouped.items():
        item: dict[str, Any] = {"model": model, "scale": scale}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            item[metric] = float(values.mean())
            item[f"{metric}_layer_sd"] = float(values.std(ddof=1))
        result.append(item)
    return sorted(
        result,
        key=lambda row: (model_order.index(row["model"]), row["scale"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    panels = [
        ("query_entropy_mean", "Query attention entropy", None),
        ("target_key_mass_ratio_mean", "Target-key mass / uniform", 1.0),
        ("q_rms_mean", "Query RMS", None),
    ]

    labels = list(dict.fromkeys(str(row["model"]) for row in rows))
    for label in labels:
        subset = [row for row in rows if row["model"] == label]
        x = np.asarray([row["scale"] for row in subset])
        for axis, (metric, ylabel, reference) in zip(axes, panels):
            y = np.asarray([row[metric] for row in subset])
            sd = np.asarray([row[f"{metric}_layer_sd"] for row in subset])
            color = COLORS.get(label)
            axis.plot(x, y, marker="o", ms=3, lw=1.5, label=label, color=color)
            axis.fill_between(x, y - sd, y + sd, alpha=0.10, color=color)
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Input scale")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22, lw=0.5)
            axis.axvline(1.0, color="#888888", ls=":", lw=0.8)
            if reference is not None:
                axis.axhline(reference, color="#777777", ls="--", lw=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        summary_path = args.output_dir / "scale_response_summary.csv"
        with summary_path.open(newline="", encoding="utf-8") as handle:
            summary = list(csv.DictReader(handle))
        for row in summary:
            for key, value in list(row.items()):
                if key != "model":
                    row[key] = float(value)
        plot_summary(args.output_dir / "scale_response_summary.pdf", summary)
        print(f"Regenerated plot in {args.output_dir}", flush=True)
        return

    models = parse_model_specs(args.model)
    if not models:
        raise ValueError("Provide at least one --model LABEL=PATH")

    all_rows: list[dict[str, Any]] = []
    for index, (label, model_path) in enumerate(models.items()):
        print(f"[{index + 1}/{len(models)}] {label}", flush=True)
        all_rows.extend(
            analyze_model(
                label=label,
                model_path=model_path,
                scales=args.scales,
                trials=args.trials,
                bank_size=args.bank_size,
                seed=args.seed,
                device=args.device,
            )
        )

    summary = aggregate(all_rows, list(models))
    write_csv(args.output_dir / "scale_response_by_layer.csv", all_rows)
    write_csv(args.output_dir / "scale_response_summary.csv", summary)
    plot_summary(args.output_dir / "scale_response_summary.pdf", summary)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "models": {
                    label: str(path) for label, path in models.items()
                },
                "seed": args.seed,
                "trials": args.trials,
                "bank_size": args.bank_size,
                "scales": args.scales,
                "device": args.device,
                "position": 0,
            },
            handle,
            indent=2,
        )
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
