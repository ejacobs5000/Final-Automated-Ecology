#!/usr/bin/env python3
"""Validate the extended BirdNET score files supplied to the visualisation team."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


HANDOFF_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CLASSES = 987
TOTAL_CLASSES = 1007
WINDOWS_PER_FILE = 20
EXPECTED_FILES = 30
ORIGINAL_THRESHOLD = 0.50
APPENDED_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HANDOFF_ROOT / "results/all_scores")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*_all_scores.npz"))
    if len(paths) != EXPECTED_FILES:
        raise RuntimeError(f"Expected {EXPECTED_FILES} score files, found {len(paths)} in {args.input_dir}")

    expected_starts = np.arange(WINDOWS_PER_FILE, dtype=float) * 3.0
    canonical_classes: list[str] | None = None
    original_events = 0
    appended_events = 0
    windows = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "start_seconds", "scores", "classes", "source_file",
                "original_report_threshold", "appended_report_threshold", "checkpoint",
            }
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(f"{path.name} is missing fields: {sorted(missing)}")
            starts = data["start_seconds"].astype(float)
            scores = data["scores"].astype(np.float32)
            classes = [str(value) for value in data["classes"]]
            source_file = str(data["source_file"].item())
            original_threshold = float(data["original_report_threshold"].item())
            appended_threshold = float(data["appended_report_threshold"].item())

        if starts.shape != (WINDOWS_PER_FILE,) or not np.array_equal(starts, expected_starts):
            raise RuntimeError(f"{path.name} has unexpected 3-second window boundaries")
        if scores.shape != (WINDOWS_PER_FILE, TOTAL_CLASSES):
            raise RuntimeError(f"{path.name} has score shape {scores.shape}, expected (20, 1007)")
        if len(classes) != TOTAL_CLASSES or len(set(classes)) != TOTAL_CLASSES:
            raise RuntimeError(f"{path.name} has malformed class labels")
        if canonical_classes is None:
            canonical_classes = classes
        elif classes != canonical_classes:
            raise RuntimeError(f"{path.name} has a different class order")
        if Path(source_file).stem != path.stem.removesuffix("_all_scores"):
            raise RuntimeError(f"{path.name} does not match source_file {source_file!r}")
        if not np.isclose(original_threshold, ORIGINAL_THRESHOLD):
            raise RuntimeError(f"{path.name} has original threshold {original_threshold}")
        if not np.isclose(appended_threshold, APPENDED_THRESHOLD):
            raise RuntimeError(f"{path.name} has appended threshold {appended_threshold}")
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise RuntimeError(f"{path.name} contains invalid sigmoid scores")

        original_events += int((scores[:, :ORIGINAL_CLASSES] >= original_threshold).sum())
        appended_events += int((scores[:, ORIGINAL_CLASSES:] >= appended_threshold).sum())
        windows += len(starts)

    if windows != 600 or original_events != 79 or appended_events != 113:
        raise RuntimeError(
            f"Unexpected delivery totals: {windows} windows, {original_events} original, "
            f"{appended_events} appended events"
        )
    print(
        "Validation passed: "
        f"{len(paths)} files, {windows} windows, {len(canonical_classes or [])} labels, "
        f"{original_events} original-head events, {appended_events} appended-head events."
    )


if __name__ == "__main__":
    main()
