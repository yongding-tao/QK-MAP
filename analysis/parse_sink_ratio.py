#!/usr/bin/env python3
"""Parse archived SinkRatio text reports into auditable CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DOMAIN_ORDER = [
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
]
SELECTED_LENGTHS = {32, 64, 128, 256, 512}
SELECTED_EPSILON = 0.1


def parse_variant_specs(specs: list[str]) -> dict[str, str]:
    variants: dict[str, str] = {}
    for spec in specs:
        raw_name, separator, label = spec.partition("=")
        if not separator or not raw_name.strip() or not label.strip():
            raise ValueError(f"Expected RAW_NAME=LABEL, got {spec!r}")
        variants[raw_name.strip()] = label.strip()
    return variants


def parse_report(
    path: Path, variants: dict[str, str]
) -> list[dict[str, object]]:
    current: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    variant_pattern = re.compile(
        r"^(\S+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s*$"
    )
    for line in path.read_text().splitlines():
        if line.startswith("target_dataset: "):
            current["domain"] = line.split(": ", 1)[1]
        elif line.startswith("seq_len: "):
            current["length"] = int(line.split(": ", 1)[1])
        elif line.startswith("num_samples: "):
            current["num_samples"] = int(line.split(": ", 1)[1])
        elif line.startswith("batch_size: "):
            current["batch_size"] = int(line.split(": ", 1)[1])
        elif line.startswith("epsilon: "):
            current["epsilon"] = float(line.split(": ", 1)[1])
        else:
            match = variant_pattern.match(line)
            if match is None:
                continue
            raw_variant, sink_ratio, spike = match.groups()
            if raw_variant not in variants:
                raise ValueError(f"Unknown variant in {path}: {raw_variant}")
            required = {"domain", "length", "num_samples", "batch_size", "epsilon"}
            if required - current.keys():
                raise ValueError(f"Incomplete block before line: {line}")
            rows.append(
                {
                    **current,
                    "variant": variants[raw_variant],
                    "sink_ratio_percent": float(sink_ratio),
                    "max_post_residual_activation": float(spike),
                    "source": path.as_posix(),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="RAW_NAME=LABEL",
        help="Map a checkpoint name in the report to a display label.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    args = parser.parse_args()

    variants = parse_variant_specs(args.variant)
    rows = [
        row
        for path in args.report
        for row in parse_report(path, variants)
    ]
    order = {label: index for index, label in enumerate(variants.values())}
    rows.sort(
        key=lambda row: (
            DOMAIN_ORDER.index(str(row["domain"])),
            int(row["length"]),
            float(row["epsilon"]),
            order[str(row["variant"])],
        )
    )
    write_csv(args.output, rows)

    selected = [
        row
        for row in rows
        if row["length"] in SELECTED_LENGTHS
        and row["epsilon"] == SELECTED_EPSILON
    ]
    write_csv(args.selected_output, selected)
    print(f"Wrote {len(rows)} raw rows to {args.output}")
    print(f"Wrote {len(selected)} epsilon=0.1 rows to {args.selected_output}")


if __name__ == "__main__":
    main()
