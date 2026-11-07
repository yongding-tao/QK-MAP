#!/usr/bin/env python3

"""Causal Q/K norm-shuffling intervention on trained 190M checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


CONTROL_CONDITIONS = ("no_hook", "clean", "reconstruct")
SHUFFLE_CONDITIONS = ("q_shuffle", "k_shuffle", "qk_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Model label and checkpoint path. Repeat once per model.",
    )
    parser.add_argument("--num-sequences", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shuffle-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Use float32 for the mixed-dtype historical HF exports.",
    )
    parser.add_argument(
        "--granularity",
        choices=["token", "head"],
        default="token",
        help=(
            "token shuffles one norm across all heads, matching the archived "
            "QK-RMSNorm axis; head shuffles a separate norm per attention head."
        ),
    )
    parser.add_argument(
        "--shuffle-axis",
        choices=["batch", "token"],
        default="batch",
        help=(
            "batch swaps norms across independent sequences at the same position "
            "and is causally safe; token is the earlier within-sequence diagnostic."
        ),
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


def load_token_sequences(
    path: Path, num_sequences: int, sequence_length: int
) -> np.ndarray:
    raw = np.memmap(path, dtype=np.uint32, mode="r")
    needed = num_sequences * sequence_length
    if needed > raw.size:
        raise ValueError(f"Need {needed} tokens, but {path} contains {raw.size}")
    return np.asarray(raw[:needed], dtype=np.int64).reshape(
        num_sequences, sequence_length
    )


def batches(tokens: np.ndarray, batch_size: int) -> Iterable[tuple[int, np.ndarray]]:
    for start in range(0, len(tokens), batch_size):
        yield start // batch_size, tokens[start : start + batch_size]


def vector_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True)


def intervention_norm(x: torch.Tensor, granularity: str) -> torch.Tensor:
    if granularity == "head":
        return vector_norm(x)
    if granularity == "token":
        return torch.linalg.vector_norm(
            x.float().flatten(start_dim=2), dim=-1, keepdim=True
        ).unsqueeze(-1)
    raise ValueError(f"Unknown granularity: {granularity}")


def recompose(
    x: torch.Tensor, new_norm: torch.Tensor, granularity: str
) -> torch.Tensor:
    old_norm = intervention_norm(x, granularity)
    direction = x / old_norm.clamp_min(1e-12).to(x.dtype)
    return direction * new_norm.to(x.dtype)


def shuffled_norm(
    norm: torch.Tensor,
    *,
    base_seed: int,
    repeat: int,
    batch_index: int,
    layer_index: int,
    channel: int,
    shuffle_axis: str,
) -> torch.Tensor:
    batch, seq_len, n_heads, _ = norm.shape
    seed = (
        base_seed
        + repeat * 10_000_000
        + batch_index * 100_000
        + layer_index * 1_000
        + channel * 100
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if shuffle_axis == "token":
        random_scores = torch.rand(
            batch, n_heads, seq_len, generator=generator, device="cpu"
        )
        indices = random_scores.argsort(dim=-1).permute(0, 2, 1).unsqueeze(-1)
        return norm.gather(1, indices.to(norm.device))
    if shuffle_axis == "batch":
        random_scores = torch.rand(
            seq_len, n_heads, batch, generator=generator, device="cpu"
        )
        indices = random_scores.argsort(dim=-1).permute(2, 0, 1).unsqueeze(-1)
        return norm.gather(0, indices.to(norm.device))
    raise ValueError(f"Unknown shuffle axis: {shuffle_axis}")


@dataclass
class CleanStats:
    query_norms: dict[int, list[np.ndarray]] = field(default_factory=dict)
    key_norms: dict[int, list[np.ndarray]] = field(default_factory=dict)
    query_token_norms: dict[int, list[np.ndarray]] = field(default_factory=dict)
    key_token_norms: dict[int, list[np.ndarray]] = field(default_factory=dict)
    abs_logits: dict[int, list[np.ndarray]] = field(default_factory=dict)
    entropies: dict[int, list[np.ndarray]] = field(default_factory=dict)

    def add(
        self,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        scale: float,
    ) -> None:
        qn = vector_norm(q).squeeze(-1)
        kn = vector_norm(k).squeeze(-1)
        self.query_norms.setdefault(layer_index, []).append(
            qn.detach().cpu().numpy().reshape(-1)
        )
        self.key_norms.setdefault(layer_index, []).append(
            kn.detach().cpu().numpy().reshape(-1)
        )
        qtn = intervention_norm(q, "token").squeeze(-1).squeeze(-1)
        ktn = intervention_norm(k, "token").squeeze(-1).squeeze(-1)
        self.query_token_norms.setdefault(layer_index, []).append(
            qtn.detach().cpu().numpy().reshape(-1)
        )
        self.key_token_norms.setdefault(layer_index, []).append(
            ktn.detach().cpu().numpy().reshape(-1)
        )

        seq_len = q.shape[1]
        positions = torch.linspace(
            max(1, seq_len // 8), seq_len - 1, steps=min(8, seq_len - 1), device=q.device
        ).long().unique()
        layer_logits: list[torch.Tensor] = []
        layer_entropies: list[torch.Tensor] = []
        for position in positions.tolist():
            scores = torch.einsum(
                "bhd,bthd->bht", q[:, position], k[:, : position + 1]
            ).float() * scale
            layer_logits.append(scores.abs().flatten())
            probs = scores.softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy / math.log(position + 1)
            layer_entropies.append(entropy.flatten())
        self.abs_logits.setdefault(layer_index, []).append(
            torch.cat(layer_logits).detach().cpu().numpy()
        )
        self.entropies.setdefault(layer_index, []).append(
            torch.cat(layer_entropies).detach().cpu().numpy()
        )

    def rows(self, model_label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        sources = {
            "q_norm_head": self.query_norms,
            "k_norm_head": self.key_norms,
            "q_norm_token": self.query_token_norms,
            "k_norm_token": self.key_token_norms,
            "abs_logit": self.abs_logits,
            "attention_entropy": self.entropies,
        }
        for metric, by_layer in sources.items():
            all_values: list[np.ndarray] = []
            for layer_index in sorted(by_layer):
                values = np.concatenate(by_layer[layer_index])
                all_values.append(values)
                rows.append(stat_row(model_label, layer_index, metric, values))
            rows.append(stat_row(model_label, "all", metric, np.concatenate(all_values)))
        return rows


def stat_row(
    model_label: str, layer: int | str, metric: str, values: np.ndarray
) -> dict[str, Any]:
    quantiles = np.quantile(values.astype(np.float64), [0.5, 0.95, 0.99, 0.999])
    return {
        "model": model_label,
        "layer": layer,
        "metric": metric,
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "p999": float(quantiles[3]),
        "max": float(np.max(values)),
    }


class Intervention:
    def __init__(
        self,
        condition: str,
        repeat: int,
        batch_index: int,
        base_seed: int,
        collect_stats: bool,
        granularity: str,
        shuffle_axis: str,
    ):
        self.condition = condition
        self.repeat = repeat
        self.batch_index = batch_index
        self.base_seed = base_seed
        self.granularity = granularity
        self.shuffle_axis = shuffle_axis
        self.collector = CleanStats() if collect_stats else None
        self.handles: list[Any] = []

    def install(self, model: torch.nn.Module) -> None:
        for layer_index, block in enumerate(model.olmo.blocks.values()):
            backend = block.attention.backend
            scale = backend.scale or (1.0 / math.sqrt(block.attention.head_dim))

            def hook(
                module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                layer_index: int = layer_index,
                scale: float = scale,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                del module
                q, k, v = args[0]
                if self.collector is not None:
                    self.collector.add(layer_index, q, k, scale)

                if self.condition == "clean":
                    return args, kwargs

                q_norm = intervention_norm(q, self.granularity)
                k_norm = intervention_norm(k, self.granularity)
                if self.condition in ("reconstruct", "q_shuffle", "qk_shuffle"):
                    replacement = q_norm
                    if self.condition in ("q_shuffle", "qk_shuffle"):
                        replacement = shuffled_norm(
                            q_norm,
                            base_seed=self.base_seed,
                            repeat=self.repeat,
                            batch_index=self.batch_index,
                            layer_index=layer_index,
                            channel=0,
                            shuffle_axis=self.shuffle_axis,
                        )
                    q = recompose(q, replacement, self.granularity)
                if self.condition in ("reconstruct", "k_shuffle", "qk_shuffle"):
                    replacement = k_norm
                    if self.condition in ("k_shuffle", "qk_shuffle"):
                        replacement = shuffled_norm(
                            k_norm,
                            base_seed=self.base_seed,
                            repeat=self.repeat,
                            batch_index=self.batch_index,
                            layer_index=layer_index,
                            channel=1,
                            shuffle_axis=self.shuffle_axis,
                        )
                    k = recompose(k, replacement, self.granularity)
                new_args = ((q, k, v),) + args[1:]
                return new_args, kwargs

            self.handles.append(
                backend.register_forward_pre_hook(hook, with_kwargs=True)
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    tokens: np.ndarray,
    *,
    condition: str,
    repeat: int,
    batch_size: int,
    base_seed: int,
    device: str,
    collect_stats: bool,
    granularity: str,
    shuffle_axis: str,
) -> tuple[np.ndarray, CleanStats | None]:
    losses: list[np.ndarray] = []
    collector = CleanStats() if collect_stats else None
    for batch_index, batch_tokens in batches(tokens, batch_size):
        intervention: Intervention | None = None
        if condition != "no_hook":
            intervention = Intervention(
                condition=condition,
                repeat=repeat,
                batch_index=batch_index,
                base_seed=base_seed,
                collect_stats=collect_stats,
                granularity=granularity,
                shuffle_axis=shuffle_axis,
            )
            intervention.install(model)

        input_ids = torch.as_tensor(batch_tokens, dtype=torch.long, device=device)
        output = model(input_ids=input_ids)
        logits = output.logits[:, :-1].float()
        labels = input_ids[:, 1:]
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(input_ids.shape[0], -1)
        losses.append(per_token.mean(dim=-1).cpu().numpy())

        if intervention is not None:
            intervention.remove()
            if collect_stats and intervention.collector is not None:
                merge_stats(collector, intervention.collector)
    return np.concatenate(losses), collector


def merge_stats(target: CleanStats | None, source: CleanStats) -> None:
    if target is None:
        return
    for name in (
        "query_norms",
        "key_norms",
        "query_token_norms",
        "key_token_norms",
        "abs_logits",
        "entropies",
    ):
        target_dict = getattr(target, name)
        for layer, chunks in getattr(source, name).items():
            target_dict.setdefault(layer, []).extend(chunks)


def bootstrap_ci(
    values: np.ndarray, samples: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(values), size=len(values))
        means[index] = values[draw].mean()
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    models = parse_model_specs(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens = load_token_sequences(args.data, args.num_sequences, args.sequence_length)

    per_sequence_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_index, (model_label, model_path) in enumerate(models.items()):
        print(f"Loading {model_label}", flush=True)
        model_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=model_dtype,
        ).to(args.device)
        model.eval()

        condition_losses: dict[tuple[str, int], np.ndarray] = {}
        schedule = [(condition, 0) for condition in CONTROL_CONDITIONS]
        schedule += [
            (condition, repeat)
            for condition in SHUFFLE_CONDITIONS
            for repeat in range(args.shuffle_repeats)
        ]
        for condition, repeat in schedule:
            print(f"  {condition} repeat={repeat}", flush=True)
            losses, collector = evaluate(
                model,
                tokens,
                condition=condition,
                repeat=repeat,
                batch_size=args.batch_size,
                base_seed=args.seed,
                device=args.device,
                collect_stats=(condition == "clean" and repeat == 0),
                granularity=args.granularity,
                shuffle_axis=args.shuffle_axis,
            )
            condition_losses[(condition, repeat)] = losses
            for sequence_index, loss in enumerate(losses):
                per_sequence_rows.append(
                    {
                        "model": model_label,
                        "condition": condition,
                        "repeat": repeat,
                        "sequence_index": sequence_index,
                        "ce": float(loss),
                    }
                )
            if collector is not None:
                stats_rows.extend(collector.rows(model_label))

        reference = condition_losses[("no_hook", 0)]
        clean_delta = condition_losses[("clean", 0)] - reference
        identity_max = float(np.max(np.abs(clean_delta)))
        if identity_max > 1e-7:
            raise RuntimeError(
                f"{model_label}: clean hook changed CE; max abs delta={identity_max}"
            )

        for condition in CONTROL_CONDITIONS + SHUFFLE_CONDITIONS:
            repeats = (
                [0]
                if condition in CONTROL_CONDITIONS
                else list(range(args.shuffle_repeats))
            )
            deltas = np.stack(
                [condition_losses[(condition, repeat)] - reference for repeat in repeats]
            ).mean(axis=0)
            mean, ci_low, ci_high = bootstrap_ci(
                deltas,
                samples=args.bootstrap_samples,
                seed=args.seed + model_index * 100 + list(CONTROL_CONDITIONS + SHUFFLE_CONDITIONS).index(condition),
            )
            summary_rows.append(
                {
                    "model": model_label,
                    "condition": condition,
                    "num_sequences": len(deltas),
                    "repeats": len(repeats),
                    "clean_ce": float(reference.mean()),
                    "delta_ce_mean": mean,
                    "delta_ce_ci_low": ci_low,
                    "delta_ce_ci_high": ci_high,
                    "delta_ce_max_abs": float(np.max(np.abs(deltas))),
                }
            )

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "per_sequence_ce.csv", per_sequence_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "clean_attention_stats.csv", stats_rows)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "models": {
                    label: str(path) for label, path in models.items()
                },
                "data": str(args.data),
                "num_sequences": args.num_sequences,
                "sequence_length": args.sequence_length,
                "batch_size": args.batch_size,
                "shuffle_repeats": args.shuffle_repeats,
                "seed": args.seed,
                "bootstrap_samples": args.bootstrap_samples,
                "device": args.device,
                "dtype": args.dtype,
                "granularity": args.granularity,
                "shuffle_axis": args.shuffle_axis,
                "intervention_point": "after QK normalization and RoPE, before attention backend",
            },
            handle,
            indent=2,
        )
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
