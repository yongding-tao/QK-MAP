#!/usr/bin/env python3
"""Summarize OLMES task outputs using each task's primary metric."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


# BoolQ is evaluated but is not included in the nine-task paper average.
PAPER_TASKS = [
    "arc_easy",
    "piqa",
    "sciq",
    "truthfulqa",
    "arc_challenge",
    "copa",
    "hellaswag",
    "openbookqa",
    "winogrande",
]
EXPECTED_TASKS = set(PAPER_TASKS) | {"boolq"}


def parse_run_specs(specs: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError(f"Expected LABEL=METRICS_ALL_JSONL, got {spec!r}")
        runs[label.strip()] = Path(raw_path).expanduser()
    return runs


def load_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            task = item["task_name"]
            if task in rows:
                raise RuntimeError(f"Duplicate task {task} in {path}")
            rows[task] = item
    if set(rows) != EXPECTED_TASKS:
        raise RuntimeError(f"Unexpected task set in {path}: {sorted(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=METRICS_ALL_JSONL",
        help="A method label and its OLMES metrics-all JSONL file.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = parse_run_specs(args.run)
    rows: list[dict[str, object]] = []
    protocol_reference: dict[str, tuple[object, ...]] = {}

    for label, path in runs.items():
        tasks = load_jsonl(path)
        scores = {
            task: float(tasks[task]["metrics"]["primary_score"]) * 100
            for task in PAPER_TASKS
        }
        rows.append(
            {
                "variant": label,
                **{task: round(score, 4) for task, score in scores.items()},
                "avg_9_tasks": round(statistics.mean(scores.values()), 4),
                "boolq_not_in_avg": round(
                    float(tasks["boolq"]["metrics"]["primary_score"]) * 100,
                    4,
                ),
            }
        )

        for task, item in tasks.items():
            task_config = item["task_config"]
            protocol = (
                task_config["primary_metric"],
                int(item["num_instances"]),
                item["task_hash"],
                task_config["split"],
                task_config["limit"],
                int(task_config["num_shots"]),
            )
            if task in protocol_reference and protocol_reference[task] != protocol:
                raise RuntimeError(
                    f"Evaluation protocol differs for {task} in {path}"
                )
            protocol_reference[task] = protocol

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
