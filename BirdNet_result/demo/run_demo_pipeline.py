#!/usr/bin/env python3
"""Denoise and score the five versioned raw-audio demo recordings.

This invokes the handoff's normal raw-audio deployment wrapper with its
explicit subset option. It performs inference only and does not retrain or
modify either model head.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parent
HANDOFF_ROOT = DEMO_ROOT.parent
RAW_AUDIO_DIR = DEMO_ROOT / "raw_audio"
EXPECTED_FILES = (
    "20220615_000002_602.flac",
    "20220615_122543_200.flac",
    "20220615_122543_324.flac",
    "20220615_122543_406.flac",
    "20220615_122543_644.flac",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Exact frozen BirdNET-V1 snapshot; see artifacts/MODEL_SETUP.md",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEMO_ROOT / "output" / "raw_to_scores",
        help="New directory for generated cleaned audio, scores, and the denoising report",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [name for name in EXPECTED_FILES if not (RAW_AUDIO_DIR / name).is_file()]
    unexpected = sorted(
        path.name for path in RAW_AUDIO_DIR.glob("*.flac") if path.name not in EXPECTED_FILES
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Demo raw audio is incomplete or changed; missing={missing}, unexpected={unexpected}"
        )
    if not args.snapshot.is_file():
        raise FileNotFoundError(f"BirdNET snapshot not found: {args.snapshot}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise FileExistsError(f"Work directory is not empty: {args.work_dir}. Choose a new --work-dir.")

    clean_dir = args.work_dir / "cleaned"
    score_dir = args.work_dir / "scores"
    command = [
        sys.executable,
        str(HANDOFF_ROOT / "code" / "run_raw_to_scores.py"),
        "--raw-input-dir",
        str(RAW_AUDIO_DIR),
        "--clean-output-dir",
        str(clean_dir),
        "--score-output-dir",
        str(score_dir),
        "--denoise-report",
        str(args.work_dir / "denoise_report.csv"),
        "--snapshot",
        str(args.snapshot),
        "--allow-subset",
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--threads",
        str(args.threads),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"Demo raw-to-score pipeline completed. Results: {args.work_dir}")


if __name__ == "__main__":
    main()
