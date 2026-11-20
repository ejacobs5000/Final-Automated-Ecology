#!/usr/bin/env python3
"""Apply the frozen BirdNET-V1 + 20-class appended head to cleaned Denoising audio.

The 987 original BirdNET classes are evaluated from their unchanged frozen
weights.  The final 20 output columns come from the separately trained,
checkpointed 1x1 convolution.  Original and appended scores have separate
reporting thresholds: 0.50 for the original zero-shot classes and the 0.10
validation-selected threshold for the appended pilot classes.

The input recordings have no species annotations, so every output is a model
candidate rather than a ground-truth label or accuracy estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import resample_poly


WORKSPACE = Path(__file__).resolve().parent
HANDOFF_ROOT = WORKSPACE.parent
DEMO_ROOT = HANDOFF_ROOT / "third_party" / "BirdNET-V1"
FINETUNE_SOURCE = WORKSPACE / "finetune_append20.py"
DEFAULT_INPUT = HANDOFF_ROOT / "data" / "cleaned"
DEFAULT_CHECKPOINT = HANDOFF_ROOT / "artifacts/append20_head.pt"
DEFAULT_REFERENCE = HANDOFF_ROOT / "results" / "zero_shot_reference"
DEFAULT_SNAPSHOT = HANDOFF_ROOT / "artifacts" / "BirdNET_Soundscape_Model.pkl"
ORIGINAL_CLASSES = 987
APPENDED_CLASSES = 20
FEATURE_LAYER = 153
ORIGINAL_THRESHOLD = 0.50
REGRESSION_TOLERANCE = 1e-6


def import_soundfile() -> Any:
    """Import the standard SoundFile dependency used for FLAC input."""
    try:
        import soundfile as soundfile  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FLAC inference requires SoundFile. Install the package dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return soundfile


def load_finetune_source() -> Any:
    """Import only the project helpers needed to reproduce its exact model."""
    spec = importlib.util.spec_from_file_location("course_finetune_append20", FINETUNE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import fine-tuning source: {FINETUNE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz_write(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def split_label(label: str) -> tuple[str, str]:
    scientific, separator, common = label.partition("_")
    return scientific, common if separator else scientific


def read_flac(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    soundfile = import_soundfile()
    audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono or multichannel audio, got {audio.shape} for {path.name}")
    if rate != target_rate:
        divisor = int(np.gcd(rate, target_rate))
        audio = resample_poly(audio, target_rate // divisor, rate // divisor).astype(np.float32)
        rate = target_rate
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite samples in {path.name}")
    return audio, rate


def load_extended_model(
    finetune: Any, checkpoint_path: Path, snapshot_path: Path, device: torch.device
) -> tuple[Any, Any, dict[str, Any], list[str], float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "format_version": 1,
        "original_class_count": ORIGINAL_CLASSES,
        "new_class_count": APPENDED_CLASSES,
        "expanded_class_count": ORIGINAL_CLASSES + APPENDED_CLASSES,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Unexpected checkpoint {key}: {checkpoint.get(key)!r}")
    if checkpoint.get("source_snapshot_sha256") != sha256(snapshot_path):
        raise RuntimeError("Checkpoint was trained against a different BirdNET snapshot")
    appended_labels = list(checkpoint.get("new_class_labels", []))
    if len(appended_labels) != APPENDED_CLASSES or len(set(appended_labels)) != APPENDED_CLASSES:
        raise RuntimeError("Checkpoint does not contain 20 unique appended labels")

    model, snapshot = finetune.load_frozen_birdnet(DEMO_ROOT, snapshot_path)
    model.to(device)
    head = finetune.AppendedHead(model).to(device)
    head.new_head.load_state_dict(checkpoint["new_head_state_dict"])
    head.eval()
    labels = list(model.classes) + appended_labels
    if len(labels) != ORIGINAL_CLASSES + APPENDED_CLASSES:
        raise RuntimeError("Combined class list has an unexpected length")
    threshold = float(checkpoint["selected_threshold"])
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(f"Invalid appended threshold: {threshold}")
    return model, head, snapshot, labels, threshold


def infer_file(
    path: Path,
    finetune: Any,
    model: Any,
    head: Any,
    snapshot: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Produce all 1,007 scores over complete, non-overlapping 3-s windows."""
    from birdnet_demo import make_spectrogram

    target_rate = int(snapshot["sample_rate"])
    audio, rate = read_flac(path, target_rate)
    width = int(float(snapshot["spec_length"]) * rate)
    starts = np.arange(0, len(audio) - width + 1, width, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError(f"{path.name} is shorter than one model window")

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            specs = np.stack(
                [make_spectrogram(audio[start : start + width], snapshot) for start in batch_starts]
            )
            features = finetune.forward_until(
                model, torch.from_numpy(specs[:, None]).to(device), FEATURE_LAYER
            )
            outputs.append(head.all_scores(features).cpu().numpy())
    scores = np.concatenate(outputs, axis=0)
    expected_shape = (len(starts), ORIGINAL_CLASSES + APPENDED_CLASSES)
    if scores.shape != expected_shape:
        raise ValueError(f"Unexpected score shape {scores.shape} for {path.name}; expected {expected_shape}")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError(f"Invalid model scores for {path.name}")
    return starts / rate, scores, len(audio) / rate


def validate_original_scores(
    path: Path,
    starts: np.ndarray,
    scores: np.ndarray,
    original_classes: list[str],
    reference_dir: Path,
) -> dict[str, Any] | None:
    """Compare the untouched 987 columns when a previous zero-shot file exists."""
    reference_path = reference_dir / f"{path.stem}_all_scores.npz"
    if not reference_path.is_file():
        return None
    with np.load(reference_path, allow_pickle=False) as reference:
        old_starts = reference["start_seconds"].astype(float)
        old_scores = reference["scores"].astype(np.float32)
        old_classes = [str(value) for value in reference["classes"]]
    if not np.array_equal(starts, old_starts):
        raise RuntimeError(f"Window boundaries differ from zero-shot reference for {path.name}")
    if old_scores.shape != (len(starts), ORIGINAL_CLASSES):
        raise RuntimeError(f"Malformed zero-shot reference: {reference_path}")
    if old_classes != original_classes:
        raise RuntimeError(f"Class order differs from zero-shot reference for {path.name}")
    max_abs_difference = float(np.max(np.abs(scores[:, :ORIGINAL_CLASSES] - old_scores)))
    if max_abs_difference > REGRESSION_TOLERANCE:
        raise RuntimeError(
            f"Original scores changed for {path.name}: max abs difference {max_abs_difference:.3g}"
        )
    return {
        "filename": path.name,
        "reference": str(reference_path),
        "max_abs_difference": max_abs_difference,
        "class_order_verified": True,
    }


def load_existing_scores(
    path: Path,
    score_path: Path,
    classes: list[str],
    snapshot: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Reuse a complete prior output only when it matches this exact model layout."""
    if not score_path.is_file():
        return None
    with np.load(score_path, allow_pickle=False) as saved:
        starts = saved["start_seconds"].astype(float)
        scores = saved["scores"].astype(np.float32)
        saved_classes = [str(value) for value in saved["classes"]]
    expected_starts = np.arange(20, dtype=float) * float(snapshot["spec_length"])
    expected_shape = (20, ORIGINAL_CLASSES + APPENDED_CLASSES)
    if not np.array_equal(starts, expected_starts):
        raise RuntimeError(f"Existing output has unexpected window boundaries: {score_path}")
    if scores.shape != expected_shape or saved_classes != classes:
        raise RuntimeError(f"Existing output has a different model layout: {score_path}")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise RuntimeError(f"Existing output contains invalid scores: {score_path}")
    return starts, scores, float(starts[-1] + float(snapshot["spec_length"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=HANDOFF_ROOT / "results/recomputed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--limit", type=int, help="Process only the first N files for a smoke test")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "Allow a non-empty input directory with fewer or more than the delivered 30 files. "
            "Use only for an intentionally selected demo or smoke-test subset."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated per-file 1,007-score outputs already in --output-dir",
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        help="Process these exact FLAC basenames (useful for a reference regression smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no CUDA device is available")
    paths = sorted(args.input_dir.glob("*.flac"))
    if not paths:
        raise RuntimeError(f"No FLAC files found in {args.input_dir}")
    if len(paths) != 30 and not args.allow_subset:
        raise RuntimeError(f"Expected 30 FLAC files in {args.input_dir}, found {len(paths)}")
    if args.filenames is not None:
        if args.limit is not None:
            raise ValueError("Use either --filenames or --limit, not both")
        requested = set(args.filenames)
        available = {path.name for path in paths}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested files are absent from {args.input_dir}: {missing}")
        paths = [path for path in paths if path.name in requested]
    if args.limit is not None:
        paths = paths[: args.limit]
        if not paths:
            raise ValueError("--limit selected no files")

    finetune = load_finetune_source()
    finetune.add_demo_root(DEMO_ROOT)
    from birdnet_demo import __file__ as birdnet_adapter_path

    torch.set_num_threads(args.threads)
    model, head, snapshot, classes, appended_threshold = load_extended_model(
        finetune, args.checkpoint, args.snapshot, device
    )
    score_dir = args.output_dir / "all_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    class_array = np.asarray(classes, dtype=str)
    candidate_rows: list[dict[str, Any]] = []
    appended_summary_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    appended_scores_by_class: list[list[float]] = [[] for _ in range(APPENDED_CLASSES)]
    appended_events_by_class = np.zeros(APPENDED_CLASSES, dtype=int)
    appended_files_by_class: list[set[str]] = [set() for _ in range(APPENDED_CLASSES)]
    total_windows = 0
    all_appended_events = 0
    all_original_events = 0
    started = time.perf_counter()

    for file_index, path in enumerate(paths, start=1):
        file_started = time.perf_counter()
        score_path = score_dir / f"{path.stem}_all_scores.npz"
        cached = (
            load_existing_scores(path, score_path, classes, snapshot) if args.resume else None
        )
        if cached is None:
            starts, scores, duration = infer_file(
                path, finetune, model, head, snapshot, args.batch_size, device
            )
            source = "inferred"
        else:
            starts, scores, duration = cached
            source = "reused"
        if duration != 60.0 or len(starts) != 20:
            raise RuntimeError(
                f"Expected 60 seconds / 20 windows for {path.name}; got {duration:.6g} / {len(starts)}"
            )
        regression = validate_original_scores(
            path, starts, scores, classes[:ORIGINAL_CLASSES], args.reference_dir
        )
        if regression is not None:
            regression_rows.append(regression)
        if source == "inferred":
            atomic_npz_write(
                score_path,
                start_seconds=starts,
                scores=scores.astype(np.float32),
                classes=class_array,
                source_file=path.name,
                original_report_threshold=np.float32(ORIGINAL_THRESHOLD),
                appended_report_threshold=np.float32(appended_threshold),
                checkpoint=str(args.checkpoint),
            )

        original_mask = scores[:, :ORIGINAL_CLASSES] >= ORIGINAL_THRESHOLD
        appended_mask = scores[:, ORIGINAL_CLASSES:] >= appended_threshold
        all_original_events += int(original_mask.sum())
        all_appended_events += int(appended_mask.sum())
        total_windows += len(starts)
        for window_index, start in enumerate(starts):
            indices = np.flatnonzero(
                np.concatenate((original_mask[window_index], appended_mask[window_index]))
            )
            for class_index in indices:
                scientific, common = split_label(classes[int(class_index)])
                candidate_rows.append(
                    {
                        "filename": path.name,
                        "window_index": window_index,
                        "start_s": f"{float(start):.1f}",
                        "end_s": f"{float(start + float(snapshot['spec_length'])):.1f}",
                        "head": "original_zero_shot" if class_index < ORIGINAL_CLASSES else "appended_pilot",
                        "class_index": int(class_index),
                        "scientific_name": scientific,
                        "common_name": common,
                        "score": f"{float(scores[window_index, class_index]):.8f}",
                        "reporting_threshold": f"{ORIGINAL_THRESHOLD if class_index < ORIGINAL_CLASSES else appended_threshold:.2f}",
                    }
                )
        for appended_index in range(APPENDED_CLASSES):
            class_scores = scores[:, ORIGINAL_CLASSES + appended_index]
            appended_scores_by_class[appended_index].extend(float(value) for value in class_scores)
            detected = class_scores >= appended_threshold
            appended_events_by_class[appended_index] += int(detected.sum())
            if detected.any():
                appended_files_by_class[appended_index].add(path.name)
        print(
            f"[{file_index}/{len(paths)}] {path.name}: {len(starts)} windows, "
            f"{int(appended_mask.sum())} appended events ({source}) in "
            f"{time.perf_counter() - file_started:.1f}s",
            flush=True,
        )

    for appended_index, label in enumerate(classes[ORIGINAL_CLASSES:]):
        values = np.asarray(appended_scores_by_class[appended_index], dtype=float)
        peak_index = int(values.argmax())
        scientific, common = split_label(label)
        appended_summary_rows.append(
            {
                "class_index": ORIGINAL_CLASSES + appended_index,
                "scientific_name": scientific,
                "common_name": common,
                "reporting_threshold": f"{appended_threshold:.2f}",
                "events_at_or_above_threshold": int(appended_events_by_class[appended_index]),
                "files_with_event": len(appended_files_by_class[appended_index]),
                "peak_score": f"{float(values.max()):.8f}",
                "mean_score": f"{float(values.mean()):.8f}",
                "peak_global_window_index": peak_index,
            }
        )

    candidate_rows.sort(key=lambda row: (row["filename"], row["window_index"], row["class_index"]))
    appended_summary_rows.sort(key=lambda row: (-int(row["events_at_or_above_threshold"]), -float(row["peak_score"])))
    write_csv(
        args.output_dir / "candidate_detections.csv",
        [
            "filename", "window_index", "start_s", "end_s", "head", "class_index",
            "scientific_name", "common_name", "score", "reporting_threshold",
        ],
        candidate_rows,
    )
    write_csv(
        args.output_dir / "appended_class_summary.csv",
        [
            "class_index", "scientific_name", "common_name", "reporting_threshold",
            "events_at_or_above_threshold", "files_with_event", "peak_score", "mean_score",
            "peak_global_window_index",
        ],
        appended_summary_rows,
    )
    write_csv(
        args.output_dir / "original_score_regression.csv",
        ["filename", "reference", "max_abs_difference", "class_order_verified"],
        regression_rows,
    )
    report = {
        "status": "completed",
        "purpose": "descriptive candidate inference on unlabeled Denoising recordings",
        "input": {
            "directory": str(args.input_dir),
            "files": len(paths),
            "windows": total_windows,
            "window_seconds": float(snapshot["spec_length"]),
        },
        "architecture": {
            "original_classes": ORIGINAL_CLASSES,
            "appended_classes": APPENDED_CLASSES,
            "total_classes": len(classes),
            "original_head": "unchanged frozen BirdNET-V1 checkpoint",
            "appended_head": str(args.checkpoint),
            "feature_layer": FEATURE_LAYER,
        },
        "thresholds": {
            "original_zero_shot": ORIGINAL_THRESHOLD,
            "appended_pilot": appended_threshold,
            "note": "The thresholds are head-specific and should not be treated as calibrated probabilities.",
        },
        "detections": {
            "original_events": all_original_events,
            "appended_events": all_appended_events,
            "total_events": len(candidate_rows),
            "appended_classes_with_events": int((appended_events_by_class > 0).sum()),
        },
        "original_score_regression": {
            "reference_files_checked": len(regression_rows),
            "tolerance": REGRESSION_TOLERANCE,
            "max_abs_difference": max((row["max_abs_difference"] for row in regression_rows), default=None),
        },
        "provenance": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": sha256(args.snapshot),
            "inference_source": str(Path(__file__).resolve()),
            "fine_tuning_source": str(FINETUNE_SOURCE),
            "birdnet_adapter": str(Path(birdnet_adapter_path).resolve()),
            "device": str(device),
        },
        "caveat": (
            "Denoising/cleaned_v4 has no species ground-truth labels. These are candidate detections only; "
            "the pilot's held-out metrics do not establish accuracy on these recordings."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json_write(args.output_dir / "run_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
