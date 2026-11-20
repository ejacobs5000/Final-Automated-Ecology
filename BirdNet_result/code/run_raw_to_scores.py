#!/usr/bin/env python3
"""Run the deployment pipeline: raw FLAC -> denoised FLAC -> BirdNET scores.

This command performs inference only.  It never trains or changes either the
frozen BirdNET base model or the appended 20-class checkpoint.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


HANDOFF_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = HANDOFF_ROOT / "code"
EXPECTED_FILES = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-input-dir", type=Path, required=True)
    parser.add_argument("--clean-output-dir", type=Path, required=True)
    parser.add_argument("--score-output-dir", type=Path, required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=HANDOFF_ROOT / "artifacts/BirdNET_Soundscape_Model.pkl",
        help="Exact frozen 987-class BirdNET-V1 snapshot; see artifacts/MODEL_SETUP.md",
    )
    parser.add_argument("--denoise-report", type=Path)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for both stages")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "Allow a non-empty input directory with fewer or more than the delivered 30 files. "
            "Use only for an intentionally selected demo or smoke-test subset."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the two commands without running them")
    return parser.parse_args()


def display(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def main() -> None:
    args = parse_args()
    raw_paths = sorted(args.raw_input_dir.glob("*.flac"))
    if not raw_paths:
        raise RuntimeError(f"No FLAC files found in {args.raw_input_dir}")
    if len(raw_paths) != EXPECTED_FILES and not args.allow_subset:
        raise RuntimeError(
            f"This delivered inference workflow expects {EXPECTED_FILES} raw FLAC files; "
            f"found {len(raw_paths)} in {args.raw_input_dir}"
        )
    if args.clean_output_dir.resolve() == args.raw_input_dir.resolve():
        raise ValueError("--clean-output-dir must differ from --raw-input-dir")
    if args.score_output_dir.resolve() in {
        args.raw_input_dir.resolve(), args.clean_output_dir.resolve()
    }:
        raise ValueError("--score-output-dir must be a separate directory")
    for name, directory, pattern in (
        ("clean", args.clean_output_dir, "*.flac"),
        ("score", args.score_output_dir, "*_all_scores.npz"),
    ):
        if directory.exists() and any(directory.glob(pattern)):
            raise FileExistsError(
                f"{name.capitalize()} output directory already contains {pattern}: {directory}. "
                "Use a fresh output directory to avoid mixing runs."
            )

    report_path = args.denoise_report or args.clean_output_dir / "denoise_report.csv"
    denoise_command = [
        args.python,
        str(CODE_ROOT / "denoise_pipeline.py"),
        "--input_dir",
        str(args.raw_input_dir),
        "--output_dir",
        str(args.clean_output_dir),
        "--report_csv",
        str(report_path),
    ]
    inference_command = [
        args.python,
        str(CODE_ROOT / "run_appended_inference.py"),
        "--input-dir",
        str(args.clean_output_dir),
        "--checkpoint",
        str(HANDOFF_ROOT / "artifacts/append20_head.pt"),
        "--snapshot",
        str(args.snapshot),
        "--output-dir",
        str(args.score_output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--threads",
        str(args.threads),
    ]
    if args.allow_subset:
        inference_command.append("--allow-subset")
    print("Denoising command:\n  " + display(denoise_command))
    print("Inference command:\n  " + display(inference_command))
    if args.dry_run:
        return
    subprocess.run(denoise_command, check=True)
    subprocess.run(inference_command, check=True)
    print(f"Completed raw-audio-to-score deployment in {args.score_output_dir}")


if __name__ == "__main__":
    main()
