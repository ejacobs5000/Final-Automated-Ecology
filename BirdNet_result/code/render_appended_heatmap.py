#!/usr/bin/env python3
"""Render fine-tuned appended-label scores for the five prior zero-shot files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parent
HANDOFF_ROOT = WORKSPACE.parent
DEFAULT_INPUT = HANDOFF_ROOT / "results/all_scores"
SELECTION = [
    "20220615_000002_602_v4.flac",
    "20220615_122543_200_v4.flac",
    "20220615_122543_324_v4.flac",
    "20220615_122543_406_v4.flac",
    "20220615_122543_644_v4.flac",
]
APPENDED_START = 987


def common_name(label: str) -> str:
    return label.split("_", 1)[1] if "_" in label else label


def darken(rgb: np.ndarray, amount: float = 0.68) -> tuple[float, float, float]:
    return tuple(np.clip(rgb * amount, 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=HANDOFF_ROOT / "examples/appended_head_five_file_heatmap.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    for filename in SELECTION:
        path = args.input_dir / f"{Path(filename).stem}_all_scores.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing combined scores: {path}")
        with np.load(path, allow_pickle=False) as data:
            scores = data["scores"].astype(float)
            classes = [str(value) for value in data["classes"]]
            records.append(
                {
                    "filename": filename,
                    "starts": data["start_seconds"].astype(float),
                    "scores": scores[:, APPENDED_START:],
                    "classes": classes[APPENDED_START:],
                    "threshold": float(data["appended_report_threshold"].item()),
                }
            )
    classes = records[0]["classes"]
    assert isinstance(classes, list)
    threshold = float(records[0]["threshold"])
    if any(record["classes"] != classes or record["threshold"] != threshold for record in records[1:]):
        raise RuntimeError("The selected score files have inconsistent appended label metadata")

    all_scores = np.concatenate([np.asarray(record["scores"]) for record in records], axis=0)
    selected = np.flatnonzero(np.max(all_scores, axis=0) >= threshold)
    selected = np.asarray(sorted(selected, key=lambda index: common_name(classes[index]).casefold()))
    if len(selected) == 0:
        raise RuntimeError("No appended class reaches its reporting threshold in the five-file sample")
    names = [common_name(classes[index]) for index in selected]
    hues = plt.get_cmap("tab20")(np.linspace(0, 1, len(selected), endpoint=False))[:, :3]
    background = np.asarray(mcolors.to_rgb("#111827"))

    fig, axes = plt.subplots(
        len(records),
        1,
        figsize=(16.2, 20.5),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    fig.subplots_adjust(left=0.20, right=0.88, top=0.925, bottom=0.09)
    for panel_index, (axis, record) in enumerate(zip(axes, records)):
        scores = np.asarray(record["scores"])[:, selected].T
        rgb = background[None, None, :] * (1.0 - scores[:, :, None]) + hues[:, None, :] * scores[:, :, None]
        axis.imshow(rgb, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names, fontsize=9.4, fontweight="semibold")
        for tick, hue in zip(axis.get_yticklabels(), hues):
            tick.set_color(darken(hue))
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
        detected_y, detected_x = np.where(scores >= threshold)
        for y, x in zip(detected_y, detected_x):
            axis.add_patch(
                patches.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.8)
            )
        if panel_index != len(records) - 1:
            axis.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Consecutive 3-second window (seconds)", fontsize=12)
    events = int(np.sum(all_scores[:, selected] >= threshold))
    fig.suptitle("Fine-tuned appended BirdNET labels across the prior five-recording sample", fontsize=19, fontweight="bold", y=0.975)
    fig.text(
        0.20,
        0.945,
        f"Only the 20 added pilot classes are shown • fixed hue per species • white outlines mark score ≥ {threshold:.2f}",
        fontsize=11.5,
        color="#253047",
    )
    fig.text(
        0.905,
        0.53,
        "How to read\n\nHue = appended species\nIntensity = score\nDark = 0\nFull color = 1",
        fontsize=10.8,
        va="center",
        ha="left",
        color="#202A3A",
        linespacing=1.35,
    )
    fig.text(
        0.905,
        0.15,
        (
            "Summary\n"
            "5 recordings\n"
            "100 windows\n"
            f"{len(selected)} appended species\n"
            f"{events} score ≥ {threshold:.2f} events\n\n"
            "Unlabeled audio: candidates only,\nnot ground truth or accuracy."
        ),
        fontsize=10.4,
        va="top",
        ha="left",
        color="#202A3A",
        linespacing=1.3,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Rendered {args.output} with {len(selected)} appended labels and {events} events")


if __name__ == "__main__":
    main()
