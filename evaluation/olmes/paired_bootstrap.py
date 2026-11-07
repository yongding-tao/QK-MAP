#!/usr/bin/env python3
"""Paired bootstrap for the QK-MAP minus QK-Norm OLMES difference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TASKS = {
    "arc_easy": "acc_uncond",
    "arc_challenge": "acc_uncond",
    "hellaswag": "acc_per_token",
    "openbookqa": "acc_uncond",
    "piqa": "acc_per_token",
    "sciq": "acc_raw",
    "winogrande": "acc_per_token",
    "truthfulqa": "mc2",
    "copa": "acc_raw",
}


def task_name(path: Path) -> str | None:
    for task in TASKS:
        if f"-{task}-predictions" in path.name:
            return task
    return None


def read_scores(path: Path, metric: str) -> tuple[list[str], np.ndarray]:
    ids = []
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.append(str(row.get("native_id", row["doc_id"])))
            values.append(float(row["metrics"][metric]) * 100)
    return ids, np.asarray(values, dtype=np.float64)


def bootstrap_means(
    differences: np.ndarray,
    reps: int,
    rng: np.random.Generator,
    batch_size: int = 500,
) -> np.ndarray:
    output = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, batch_size):
        stop = min(start + batch_size, reps)
        indices = rng.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        output[start:stop] = differences[indices].mean(axis=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qknorm-dir", type=Path, required=True)
    parser.add_argument("--qkmap-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    qk_files = {
        name: path
        for path in args.qknorm_dir.glob("*.jsonl")
        if (name := task_name(path)) is not None
    }
    qkmap_files = {
        name: path
        for path in args.qkmap_dir.glob("*.jsonl")
        if (name := task_name(path)) is not None
    }
    rng = np.random.default_rng(args.seed)
    task_bootstraps = {}
    rows = []
    for task, metric in TASKS.items():
        qk_ids, qk = read_scores(qk_files[task], metric)
        qkmap_ids, qkmap = read_scores(qkmap_files[task], metric)
        if qk_ids != qkmap_ids:
            raise RuntimeError(f"Prediction order/IDs differ for {task}")
        differences = qkmap - qk
        samples = bootstrap_means(differences, args.reps, rng)
        task_bootstraps[task] = samples
        rows.append(
            {
                "task": task,
                "metric": metric,
                "n": len(differences),
                "qkmap_minus_qknorm_points": round(float(differences.mean()), 4),
                "ci95_low": round(float(np.quantile(samples, 0.025)), 4),
                "ci95_high": round(float(np.quantile(samples, 0.975)), 4),
                "prob_qkmap_gt_qknorm": round(float((samples > 0).mean()), 4),
            }
        )

    avg9_samples = np.mean(np.stack(list(task_bootstraps.values())), axis=0)
    observed = sum(
        row["qkmap_minus_qknorm_points"] for row in rows
    ) / len(rows)
    rows.append(
        {
            "task": "unweighted_avg9",
            "metric": "mean of nine task primary metrics",
            "n": sum(int(row["n"]) for row in rows),
            "qkmap_minus_qknorm_points": round(float(observed), 4),
            "ci95_low": round(float(np.quantile(avg9_samples, 0.025)), 4),
            "ci95_high": round(float(np.quantile(avg9_samples, 0.975)), 4),
            "prob_qkmap_gt_qknorm": round(float((avg9_samples > 0).mean()), 4),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['task']:16s} diff={row['qkmap_minus_qknorm_points']:+.4f} "
            f"CI=[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] "
            f"P(>0)={row['prob_qkmap_gt_qknorm']:.4f}"
        )


if __name__ == "__main__":
    main()
