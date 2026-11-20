#!/usr/bin/env python3
"""Run the extended BirdNET model on the five versioned demo recordings.

The frozen 987-class BirdNET-V1 snapshot is not distributed in this package.
Pass the exact approved snapshot path with ``--snapshot``; the underlying
inference program verifies its checksum before it loads the model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parent
HANDOFF_ROOT = DEMO_ROOT.parent
AUDIO_DIR = DEMO_ROOT / "audio"
EXPECTED_FILES = (
    "20220615_000002_602_v4.flac",
    "20220615_122543_200_v4.flac",
    "20220615_122543_324_v4.flac",
    "20220615_122543_406_v4.flac",
    "20220615_122543_644_v4.flac",
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
        "--output-dir",
        type=Path,
        default=DEMO_ROOT / "output",
        help="New directory for score matrices and CSV/JSON summaries",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [name for name in EXPECTED_FILES if not (AUDIO_DIR / name).is_file()]
    unexpected = sorted(path.name for path in AUDIO_DIR.glob("*.flac") if path.name not in EXPECTED_FILES)
    if missing or unexpected:
        raise RuntimeError(
            f"Demo audio is incomplete or changed; missing={missing}, unexpected={unexpected}"
        )
    if not args.snapshot.is_file():
        raise FileNotFoundError(f"BirdNET snapshot not found: {args.snapshot}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Choose a new --output-dir."
        )

    command = [
        sys.executable,
        str(HANDOFF_ROOT / "code" / "run_appended_inference.py"),
        "--input-dir",
        str(AUDIO_DIR),
        "--checkpoint",
        str(HANDOFF_ROOT / "artifacts" / "append20_head.pt"),
        "--snapshot",
        str(args.snapshot),
        "--output-dir",
        str(args.output_dir),
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
    print(f"Demo inference completed. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
