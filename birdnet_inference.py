"""
Real, on-demand BirdNET inference for one freshly uploaded audio file.

BirdNet_result/'s own command-line tools (run_raw_to_scores.py,
run_appended_inference.py) are hard-wired to a fixed 30-file delivery batch.
Their *per-file* building blocks underneath that wrapper aren't, though:
denoise_pipeline.process_file() and run_appended_inference.infer_file() each
already operate on a single file. This module imports those functions
directly and drives them against one upload at a time, instead of going
through the batch CLI.

Requires artifacts/BirdNET_Soundscape_Model.pkl in place (see
BirdNet_result/artifacts/MODEL_SETUP.md) — model_available() reports whether
it's there before anything tries to use it.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import birdnet_results

HANDOFF_ROOT = Path(__file__).parent / "BirdNet_result"
CODE_ROOT = HANDOFF_ROOT / "code"
SNAPSHOT_PATH = HANDOFF_ROOT / "artifacts" / "BirdNET_Soundscape_Model.pkl"
CHECKPOINT_PATH = HANDOFF_ROOT / "artifacts" / "append20_head.pt"
EXPECTED_SNAPSHOT_SHA256 = "979c6133147354a87f8ee62cc0d77182259014e49aa0c5eb04d99614729a839e"

ORIGINAL_THRESHOLD = 0.50

_model = None  # cache: (rai_module, finetune_module, model, head, snapshot, classes, appended_threshold, device)


def model_available() -> bool:
    return SNAPSHOT_PATH.is_file() and CHECKPOINT_PATH.is_file()


def _load_model():
    global _model
    if _model is not None:
        return _model

    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    import torch
    rai = importlib.import_module("run_appended_inference")
    import denoise_pipeline  # noqa: F401 (imported here so failures surface before model load)

    finetune = rai.load_finetune_source()
    finetune.add_demo_root(rai.DEMO_ROOT)
    device = torch.device("cpu")
    model, head, snapshot, classes, appended_threshold = rai.load_extended_model(
        finetune, CHECKPOINT_PATH, SNAPSHOT_PATH, device
    )
    _model = (rai, finetune, model, head, snapshot, classes, appended_threshold, device)
    return _model


def analyze_audio_file(upload_path: Path, base_timestamp: datetime) -> tuple[list[dict], list[dict]]:
    """Denoise + run real BirdNET inference on one uploaded recording.

    base_timestamp anchors the file's first 3-second window — a real field
    timestamp isn't available for an arbitrary upload the way it is for the
    delivered campaign's YYYYMMDD_HHMMSS filenames, so the caller supplies
    one (the project's start date) and it's used only to place the result in
    time, not as a claim about when the recording was actually made.

    Returns (detections, species_catalog_entries) in this app's shapes.
    """
    rai, finetune, model, head, snapshot, classes, appended_threshold, device = _load_model()
    import denoise_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        clean_dir = Path(tmp)
        metrics = denoise_pipeline.process_file(str(upload_path), str(clean_dir))
        cleaned_path = clean_dir / metrics["output_file"]
        starts, scores, duration = rai.infer_file(
            cleaned_path, finetune, model, head, snapshot, batch_size=8, device=device
        )

    original_classes = len(classes) - 20
    detections = []
    catalog_entries = {}
    for window_index, start_s in enumerate(starts):
        window_scores = scores[window_index]
        for class_index, score in enumerate(window_scores):
            threshold = ORIGINAL_THRESHOLD if class_index < original_classes else appended_threshold
            if score < threshold:
                continue
            scientific, common = rai.split_label(classes[class_index])
            entry = birdnet_results.catalog_entry(scientific, common)
            catalog_entries.setdefault(entry["key"], entry)
            detections.append({
                "species_key": entry["key"],
                "site": None,  # filled in by the caller, which knows the project's site
                "timestamp": (base_timestamp + timedelta(seconds=float(start_s))).isoformat(),
                "confidence": round(float(score), 4),
            })

    return detections, list(catalog_entries.values())
