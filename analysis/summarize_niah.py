#!/usr/bin/env python3
"""Summarize recovered needle-attention diagnostic JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_variant_specs(specs: list[str]) -> dict[str, str]:
    variants: dict[str, str] = {}
    for spec in specs:
        pattern, separator, label = spec.partition("=")
        if not separator or not pattern.strip() or not label.strip():
            raise ValueError(f"Expected TAG_SUBSTRING=LABEL, got {spec!r}")
        variants[pattern.strip()] = label.strip()
    return variants


def variant_from_tag(tag: str, variants: dict[str, str]) -> str:
    for pattern, label in variants.items():
        if pattern in tag:
            return label
    raise ValueError(f"Unrecognized checkpoint tag: {tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="TAG_SUBSTRING=LABEL",
    )
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--length-output", type=Path, required=True)
    args = parser.parse_args()
    variants = parse_variant_specs(args.variant)

    summary_rows: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    for root in args.root:
        for path in sorted(root.rglob("metrics.json")):
            if "needle_attention_benchmark" not in path.as_posix():
                continue
            payload = json.loads(path.read_text())
            variant = variant_from_tag(payload["checkpoint_tag"], variants)
            base = {
                "dataset": payload["dataset_name"],
                "variant": variant,
                "sample_count": payload["sample_count"],
                "mean_prompt_token_count": payload["mean_prompt_token_count"],
                "source": path.as_posix(),
            }
            for pooling in ("best_layer", "average_layer", "last_layer"):
                summary_rows.append(
                    {
                        **base,
                        "layer_pooling": pooling,
                        "snr": payload["overall"][pooling]["snr"],
                        "lift": payload["overall"][pooling]["lift"],
                    }
                )
            for length, result in payload["by_context_length"].items():
                for pooling in ("best_layer", "average_layer", "last_layer"):
                    length_rows.append(
                        {
                            **base,
                            "context_length": int(length),
                            "length_sample_count": result["sample_count"],
                            "layer_pooling": pooling,
                            "snr": result[pooling]["snr"],
                            "lift": result[pooling]["lift"],
                        }
                    )

    variant_order = {
        label: index for index, label in enumerate(variants.values())
    }
    summary_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            variant_order[str(row["variant"])],
            str(row["layer_pooling"]),
        )
    )
    length_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["context_length"]),
            variant_order[str(row["variant"])],
            str(row["layer_pooling"]),
        )
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (args.summary_output, summary_rows),
        (args.length_output, length_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
