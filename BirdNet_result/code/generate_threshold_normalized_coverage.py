#!/usr/bin/env python3
"""Render threshold-normalized BirdNET candidate-label coverage.

Raw scores remain untouched.  Only the displayed colour intensity is scaled
by its own head's reporting threshold, so a score at either threshold has full
colour intensity.  This visualization transform does not calibrate or compare
the two heads' confidence values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
HANDOFF_ROOT = HERE.parent
DEFAULT_INPUT = HANDOFF_ROOT / "results/all_scores"
DEFAULT_OUTPUT = HANDOFF_ROOT / "examples/fine_tuned_coverage_threshold_normalized.png"
SELECTION = [
    "20220615_000002_602_v4.flac",
    "20220615_122543_200_v4.flac",
    "20220615_122543_324_v4.flac",
    "20220615_122543_406_v4.flac",
    "20220615_122543_644_v4.flac",
]
ORIGINAL_CLASSES = 987


def common_name(label: str) -> str:
    return label.split("_", 1)[1] if "_" in label else label


def darken(rgb: np.ndarray, amount: float = 0.68) -> tuple[float, float, float]:
    return tuple(np.clip(rgb * amount, 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    for filename in SELECTION:
        path = args.input_dir / f"{Path(filename).stem}_all_scores.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing combined scores: {path}")
        with np.load(path, allow_pickle=False) as data:
            records.append(
                {
                    "filename": filename,
                    "starts": data["start_seconds"].astype(float),
                    "scores": data["scores"].astype(float),
                    "classes": [str(value) for value in data["classes"]],
                    "original_threshold": float(data["original_report_threshold"].item()),
                    "appended_threshold": float(data["appended_report_threshold"].item()),
                }
            )
    classes = records[0]["classes"]
    assert isinstance(classes, list)
    original_threshold = float(records[0]["original_threshold"])
    appended_threshold = float(records[0]["appended_threshold"])
    if any(
        record["classes"] != classes
        or record["original_threshold"] != original_threshold
        or record["appended_threshold"] != appended_threshold
        for record in records[1:]
    ):
        raise RuntimeError("The selected score files have inconsistent metadata")

    all_scores = np.concatenate([np.asarray(record["scores"]) for record in records], axis=0)
    original_ids = np.flatnonzero(np.max(all_scores[:, :ORIGINAL_CLASSES], axis=0) >= original_threshold)
    appended_ids = ORIGINAL_CLASSES + np.flatnonzero(
        np.max(all_scores[:, ORIGINAL_CLASSES:], axis=0) >= appended_threshold
    )
    original_ids = np.asarray(sorted(original_ids, key=lambda index: common_name(classes[index]).casefold()))
    appended_ids = np.asarray(sorted(appended_ids, key=lambda index: common_name(classes[index]).casefold()))
    selected = np.concatenate((original_ids, appended_ids))
    thresholds = np.concatenate(
        (
            np.full(len(original_ids), original_threshold),
            np.full(len(appended_ids), appended_threshold),
        )
    )
    names = [common_name(classes[index]) for index in selected]
    hues = plt.get_cmap("tab20")(np.linspace(0, 1, len(selected), endpoint=False))[:, :3]
    background = np.asarray(mcolors.to_rgb("#111827"))
    original_events = int((all_scores[:, :ORIGINAL_CLASSES] >= original_threshold).sum())
    appended_events = int((all_scores[:, ORIGINAL_CLASSES:] >= appended_threshold).sum())

    fig, axes = plt.subplots(
        len(records),
        1,
        figsize=(17.2, 22.0),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    fig.subplots_adjust(left=0.25, right=0.88, top=0.90, bottom=0.085)
    for panel_index, (axis, record) in enumerate(zip(axes, records)):
        raw_scores = np.asarray(record["scores"])[:, selected].T
        # A display value of 1 means this row reaches its own reporting
        # threshold.  Clipping preserves that common visual reference point
        # without changing any saved model score or event decision.
        display_scores = np.clip(raw_scores / thresholds[:, None], 0.0, 1.0)
        rgb = (
            background[None, None, :] * (1.0 - display_scores[:, :, None])
            + hues[:, None, :] * display_scores[:, :, None]
        )
        axis.imshow(rgb, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names, fontsize=9.25, fontweight="semibold")
        for tick, hue in zip(axis.get_yticklabels(), hues):
            tick.set_color(darken(hue))
        axis.axhline(len(original_ids) - 0.5, color="white", linewidth=1.3, alpha=0.62)
        starts = np.asarray(record["starts"])
        axis.set_xticks(np.arange(len(starts)))
        axis.set_xticklabels(
            [f"{int(start)}–{int(start + 3)}" for start in starts], rotation=45, ha="right", fontsize=9
        )
        axis.set_xticks(np.arange(-0.5, len(starts), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.45, alpha=0.23)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.tick_params(axis="y", length=0, pad=7)
        axis.tick_params(axis="x", length=0)
        axis.set_title(str(record["filename"]), loc="left", fontsize=13.5, fontweight="bold", pad=8)
        detected_y, detected_x = np.where(raw_scores >= thresholds[:, None])
        for y, x in zip(detected_y, detected_x):
            axis.add_patch(
                patches.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.75)
            )
        if panel_index != len(records) - 1:
            axis.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Consecutive 3-second window (seconds)", fontsize=12)
    fig.suptitle(
        "BirdNET candidate-label coverage: threshold-normalized display",
        fontsize=19.5,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.25,
        0.943,
        "Same five 60-second recordings • top section: original 987 zero-shot labels • bottom section: newly appended pilot labels",
        fontsize=11.3,
        color="#253047",
    )
    fig.text(
        0.25,
        0.918,
        (
            f"Original: {len(original_ids)} labels / {original_events} events (threshold ≥ {original_threshold:.2f})"
            f"     Added: +{len(appended_ids)} labels / +{appended_events} events (threshold ≥ {appended_threshold:.2f})"
            f"     Combined: {len(selected)} labels / {original_events + appended_events} events"
        ),
        fontsize=11.1,
        color="#253047",
        fontweight="semibold",
    )
    fig.text(
        0.895,
        0.53,
        "How to read\n\nHue = candidate species\nIntensity = raw score ÷ its\nhead-specific threshold\nFull colour = threshold reached\nWhite outline = candidate event\n\nHorizontal line separates\noriginal and appended labels.",
        fontsize=10.4,
        va="center",
        ha="left",
        color="#202A3A",
        linespacing=1.35,
    )
    fig.text(
        0.895,
        0.14,
        "Display normalization only:\nraw scores, thresholds, and event\ndecisions are unchanged. Brightness\nshows relative-to-threshold strength,\nnot calibrated confidence or accuracy.",
        fontsize=10.1,
        va="top",
        ha="left",
        color="#202A3A",
        linespacing=1.3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"Rendered {args.output} with {len(original_ids)} original + {len(appended_ids)} appended labels"
    )


if __name__ == "__main__":
    main()
