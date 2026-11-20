#!/usr/bin/env python3
"""Render BirdNET zero-shot scores with a fixed hue for each species."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "results" / "zero_shot_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def common_name(class_label: str) -> str:
    return class_label.split("_", 1)[1] if "_" in class_label else class_label


def darken(rgb: np.ndarray, amount: float = 0.68) -> tuple[float, float, float]:
    return tuple(np.clip(rgb * amount, 0.0, 1.0))


def main() -> None:
    args = parse_args()
    score_files = sorted(args.input_dir.glob("*_all_scores.npz"))
    if len(score_files) != 5:
        raise RuntimeError(f"Expected five score files, found {len(score_files)}")

    records: list[dict[str, object]] = []
    for path in score_files:
        with np.load(path, allow_pickle=False) as data:
            records.append(
                {
                    "filename": as_text(data["source_file"].item()),
                    "starts": data["start_seconds"].astype(float),
                    "scores": data["scores"].astype(float),
                    "classes": [as_text(x) for x in data["classes"]],
                    "threshold": float(data["report_threshold"].item()),
                }
            )

    classes = records[0]["classes"]
    assert isinstance(classes, list)
    threshold = float(records[0]["threshold"])
    for record in records[1:]:
        if record["classes"] != classes:
            raise RuntimeError("Class order differs between score files")

    all_scores = np.concatenate(
        [np.asarray(record["scores"]) for record in records], axis=0
    )
    selected = np.flatnonzero(np.max(all_scores, axis=0) >= threshold)
    selected = np.asarray(
        sorted(selected, key=lambda idx: common_name(classes[idx]).casefold())
    )
    names = [common_name(classes[idx]) for idx in selected]

    # Equal-purpose, high-chroma hues. Each row keeps the same hue in every panel.
    palette_hex = [
        "#20C997",  # teal
        "#FF922B",  # orange
        "#FA5252",  # red
        "#BE4BDB",  # purple
        "#FCC419",  # gold
        "#51CF66",  # green
        "#F783AC",  # pink
        "#339AF0",  # blue
    ]
    if len(selected) > len(palette_hex):
        hues = plt.get_cmap("hsv")(np.linspace(0, 1, len(selected), endpoint=False))[:, :3]
    else:
        hues = np.asarray([mcolors.to_rgb(c) for c in palette_hex[: len(selected)]])

    background = np.asarray(mcolors.to_rgb("#111827"))
    fig, axes = plt.subplots(
        len(records),
        1,
        figsize=(16.2, 19.5),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    fig.subplots_adjust(left=0.17, right=0.88, top=0.925, bottom=0.09)

    for panel_idx, (ax, record) in enumerate(zip(axes, records)):
        scores = np.asarray(record["scores"])[:, selected].T
        scores = np.clip(scores, 0.0, 1.0)

        # Linear alpha-style blend: 0 is the shared dark background and 1 is
        # the species' full hue. Thus color strength directly follows score.
        rgb = (
            background[None, None, :] * (1.0 - scores[:, :, None])
            + hues[:, None, :] * scores[:, :, None]
        )
        ax.imshow(rgb, vmin=0, vmax=1, interpolation="nearest", aspect="auto")

        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=10.5, fontweight="semibold")
        for tick, hue in zip(ax.get_yticklabels(), hues):
            tick.set_color(darken(hue))

        starts = np.asarray(record["starts"])
        ax.set_xticks(np.arange(len(starts)))
        ax.set_xticklabels(
            [f"{int(start)}–{int(start + 3)}" for start in starts],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_xticks(np.arange(-0.5, len(starts), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.23)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.tick_params(axis="x", length=0)

        filename = Path(str(record["filename"])).name
        ax.set_title(filename, loc="left", fontsize=14, fontweight="bold", pad=8)

        detected_y, detected_x = np.where(scores >= threshold)
        for y, x in zip(detected_y, detected_x):
            ax.add_patch(
                patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="white",
                    linewidth=1.9,
                )
            )

        if panel_idx != len(records) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Consecutive 3-second window (seconds)", fontsize=12)

    fig.suptitle(
        "BirdNET-V1 zero-shot detections across five 60-second recordings",
        fontsize=21,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.17,
        0.945,
        (
            "Fixed hue per species • color intensity scales linearly with BirdNET score "
            f"• white outlines mark score ≥ {threshold:.1f}"
        ),
        fontsize=12.5,
        color="#253047",
    )

    fig.text(
        0.905,
        0.54,
        "How to read\n\nHue = species\n(see colored row name)\n\nIntensity = score\nDark = 0\nFull color = 1",
        fontsize=11.5,
        va="center",
        ha="left",
        color="#202A3A",
        linespacing=1.35,
    )
    event_count = int(np.sum(all_scores[:, selected] >= threshold))
    fig.text(
        0.905,
        0.15,
        (
            "Summary\n"
            f"5 recordings\n"
            f"100 windows\n"
            f"{len(selected)} species\n"
            f"{event_count} score ≥ {threshold:.1f} events\n\n"
            "Scores are uncalibrated model outputs.\n"
            "Species labels are candidates, not ground truth."
        ),
        fontsize=10.8,
        va="top",
        ha="left",
        color="#202A3A",
        linespacing=1.3,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
