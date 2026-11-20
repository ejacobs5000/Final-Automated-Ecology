#!/usr/bin/env python3
"""Export the 1,007-column BirdNET score layout as a tidy CSV table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


HANDOFF_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CLASSES = 987


def parse_label(label: str) -> tuple[str, str]:
    scientific_name, separator, common_name = label.partition("_")
    return scientific_name, common_name if separator else scientific_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=HANDOFF_ROOT / "results/all_scores"
    )
    parser.add_argument(
        "--output", type=Path, default=HANDOFF_ROOT / "results/class_index_map.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_paths = sorted(args.input_dir.glob("*_all_scores.npz"))
    if not score_paths:
        raise FileNotFoundError(f"No score files found in {args.input_dir}")
    with np.load(score_paths[0], allow_pickle=False) as data:
        classes = [str(value) for value in data["classes"]]
        original_threshold = float(data["original_report_threshold"].item())
        appended_threshold = float(data["appended_report_threshold"].item())
    if len(classes) != 1007:
        raise RuntimeError(f"Expected 1,007 classes, found {len(classes)}")

    rows: list[dict[str, object]] = []
    for class_index, label in enumerate(classes):
        scientific_name, common_name = parse_label(label)
        is_original = class_index < ORIGINAL_CLASSES
        rows.append(
            {
                "class_index": class_index,
                "head": "original_zero_shot" if is_original else "appended_pilot",
                "scientific_name": scientific_name,
                "common_name": common_name,
                "label": label,
                "reporting_threshold": f"{original_threshold if is_original else appended_threshold:.2f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} class mappings to {args.output}")


if __name__ == "__main__":
    main()
