#!/usr/bin/env python3
"""Append and train 20 BirdNET-V1 classes without changing the original 987.

The supplied BirdNET-V1 snapshot is a legacy Theano/Lasagne model.  The local
``birdnet_demo`` adapter can evaluate it in PyTorch, but intentionally exposes
no trainable parameters.  This script uses its frozen feature tensor immediately
before the 987-class convolution and trains a new 20-class 1x1 convolution on
top.  Thus the original 987 scores are bit-for-bit determined by the untouched
snapshot; only the appended classes can change.

The pilot labels are recording-level iNaturalist labels, not timestamped call
annotations.  Accordingly, the 987 pre-existing labels are *unobserved*, rather
than negative, during training.  Binary cross-entropy is computed only across
the 20 appended labels.

Typical use (run from this directory):

  python finetune_append20.py prepare --run-dir runs/pilot_20x50
  python finetune_append20.py cache --run-dir runs/pilot_20x50 --batch-size 4
  python finetune_append20.py train --run-dir runs/pilot_20x50 --epochs 80

The default split is stratified by individual source WAV/Hugging Face row:
35 train, 7 validation, and 8 test recordings per species.  The mirror omits
the upstream recording IDs, so this is the strongest source-level grouping that
can be established from the available pilot metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly


PROJECT = Path(__file__).resolve().parent
HANDOFF_ROOT = PROJECT.parent
DEFAULT_MANIFEST = HANDOFF_ROOT / "data" / "training_manifest.csv"
DEFAULT_AUDIO_ROOT = HANDOFF_ROOT / "data" / "training_audio"
DEFAULT_DEMO_ROOT = HANDOFF_ROOT / "third_party" / "BirdNET-V1"
DEFAULT_SNAPSHOT = HANDOFF_ROOT / "artifacts" / "BirdNET_Soundscape_Model.pkl"
SPLITS = ("train", "validation", "test")
CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Record:
    """A downloaded recording and its newly appended class index."""

    hf_row_index: int
    species_index: int
    scientific_name: str
    common_name: str
    path: Path
    split: str

    @property
    def label(self) -> str:
        return f"{self.scientific_name}_{self.common_name}"


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_split_counts(value: str) -> dict[str, int]:
    try:
        counts = {key: int(number) for key, number in (part.split("=", 1) for part in value.split(","))}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "split counts must look like train=35,validation=7,test=8"
        ) from exc
    if set(counts) != set(SPLITS) or any(count <= 0 for count in counts.values()):
        raise argparse.ArgumentTypeError(
            "split counts must specify positive train, validation, and test values"
        )
    return counts


def stable_shuffle(rows: list[dict[str, str]], seed: int, species_index: int) -> list[dict[str, str]]:
    """Shuffle a class deterministically, without depending on Python hash salt."""
    shuffled = list(rows)
    rng = random.Random(f"birdnet-append20:{seed}:{species_index}")
    rng.shuffle(shuffled)
    return shuffled


def prepare_split(args: argparse.Namespace) -> None:
    source_rows = read_csv(args.manifest)
    if len(source_rows) != 1000:
        raise RuntimeError(f"Expected 1,000 pilot rows, found {len(source_rows)}")
    counts = args.split_counts
    expected_per_species = sum(counts.values())
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if row["split"] != "train":
            raise RuntimeError(f"Unexpected source split {row['split']!r}")
        grouped[int(row["species_index"])].append(row)
    if sorted(grouped) != list(range(20)):
        raise RuntimeError(f"Expected species indices 0..19, found {sorted(grouped)}")
    for species_index, rows in grouped.items():
        if len(rows) != expected_per_species:
            raise RuntimeError(
                f"Species {species_index} has {len(rows)} recordings; expected {expected_per_species}"
            )
        if len({row["hf_row_index"] for row in rows}) != len(rows):
            raise RuntimeError(f"Species {species_index} contains duplicate Hugging Face rows")

    destination = args.run_dir / "split_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(source_rows[0]) + ["pilot_split", "source_group"]
    assignments: list[dict[str, str]] = []
    for species_index in sorted(grouped):
        shuffled = stable_shuffle(grouped[species_index], args.seed, species_index)
        offset = 0
        for split in SPLITS:
            for row in shuffled[offset : offset + counts[split]]:
                assignment = dict(row)
                assignment["pilot_split"] = split
                # The mirror hides upstream recording IDs. Each downloaded row is
                # the independently addressable source recording available here.
                assignment["source_group"] = f"hf_row_{row['hf_row_index']}"
                assignments.append(assignment)
            offset += counts[split]
    assignments.sort(key=lambda row: (int(row["species_index"]), int(row["hf_row_index"])))
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignments)

    report = {
        "purpose": "internal held-out pilot split for appended BirdNET-V1 classes",
        "seed": args.seed,
        "source_manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "split_counts_per_species": counts,
        "total_counts": {split: counts[split] * 20 for split in SPLITS},
        "grouping": (
            "one Hugging Face row per group; the mirror omits upstream recording IDs, "
            "so duplicate-source relationships cannot be checked"
        ),
        "new_class_count": 20,
        "output_dimension_after_append": 1007,
    }
    atomic_json_write(args.run_dir / "split_report.json", report)
    print(json.dumps(report, indent=2))


def load_records(split_manifest: Path, audio_root: Path) -> list[Record]:
    rows = read_csv(split_manifest)
    if len(rows) != 1000:
        raise RuntimeError(f"Expected 1,000 split rows, found {len(rows)}")
    records: list[Record] = []
    for row in rows:
        split = row["pilot_split"]
        if split not in SPLITS:
            raise RuntimeError(f"Unexpected pilot split {split!r}")
        path = audio_root / row["download_relpath"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing downloaded WAV: {path}")
        records.append(
            Record(
                hf_row_index=int(row["hf_row_index"]),
                species_index=int(row["species_index"]),
                scientific_name=row["scientific_name"],
                common_name=row["common_name"],
                path=path,
                split=split,
            )
        )
    labels = {record.species_index: record.label for record in records}
    if sorted(labels) != list(range(20)) or len(set(labels.values())) != 20:
        raise RuntimeError("Split manifest does not contain exactly 20 unique indexed species")
    observed = Counter(record.split for record in records)
    if observed != Counter({"train": 700, "validation": 140, "test": 160}):
        raise RuntimeError(f"Unexpected split counts: {dict(observed)}")
    return records


def add_demo_root(demo_root: Path) -> None:
    root = str(demo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_frozen_birdnet(demo_root: Path, snapshot_path: Path) -> tuple[Any, dict[str, Any]]:
    add_demo_root(demo_root)
    from birdnet_demo import BirdNETV1, load_snapshot  # imported after sys.path setup

    snapshot = load_snapshot(snapshot_path)
    model = BirdNETV1(snapshot)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if len(model.classes) != 987:
        raise RuntimeError(f"Expected 987 original classes, found {len(model.classes)}")
    return model, snapshot


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot see a CUDA device")
    return device


def forward_until(model: Any, x: torch.Tensor, last_index: int) -> torch.Tensor:
    """Evaluate the inference adapter through an inclusive legacy layer index."""
    if tuple(x.shape[1:]) != (1, 64, 384):
        raise ValueError(f"Expected (batch, 1, 64, 384), got {tuple(x.shape)}")
    values: list[torch.Tensor] = []
    for index, node in enumerate(model.layers[: last_index + 1]):
        kind = type(node).__name__
        get = lambda key: getattr(model, f"n{index}_{key}", None)
        if kind == "InputLayer":
            value = x
        elif kind == "ElemwiseSumLayer":
            value = sum(
                coefficient * values[model.indices[id(parent)]]
                for coefficient, parent in zip(node.coeffs, node.input_layers)
            )
        else:
            previous = values[model.indices[id(node.input_layer)]]
            if kind == "Conv2DLayer":
                if node.nonlinearity.__name__ != "linear":
                    raise RuntimeError(f"Unexpected convolution activation at layer {index}")
                pad = tuple(size // 2 for size in node.filter_size) if node.pad == "same" else node.pad
                value = F.conv2d(
                    previous,
                    get("W"),
                    get("b"),
                    stride=node.stride,
                    padding=pad,
                    groups=node.num_groups,
                )
            elif kind in ("BatchNormDNNLayer", "BatchNormLayer"):
                if node.axes != (0, 2, 3):
                    raise RuntimeError(f"Unexpected batch-norm axes at layer {index}")
                beta, gamma, mean, inv_std = [
                    get(key)[None, :, None, None]
                    for key in ("beta", "gamma", "mean", "inv_std")
                ]
                value = (previous - mean) * (gamma * inv_std) + beta
            elif kind == "NonlinearityLayer":
                activation = node.nonlinearity.__name__
                if activation == "rectify":
                    value = F.relu(previous)
                elif activation == "sigmoid":
                    value = torch.sigmoid(previous)
                elif activation == "linear":
                    value = previous
                else:
                    raise RuntimeError(f"Unsupported activation {activation!r}")
            elif kind in ("MaxPool2DLayer", "Pool2DLayer"):
                if not node.ignore_border:
                    raise RuntimeError(f"Unexpected pool behaviour at layer {index}")
                if node.mode == "max":
                    value = F.max_pool2d(previous, node.pool_size, node.stride, node.pad)
                elif node.mode == "average_exc_pad":
                    value = F.avg_pool2d(
                        previous, node.pool_size, node.stride, node.pad, count_include_pad=False
                    )
                else:
                    raise RuntimeError(f"Unsupported pooling mode {node.mode!r}")
            elif kind == "DropoutLayer":
                value = previous
            elif kind == "GlobalPoolLayer":
                if node.pool_function.__name__ != "logmeanexp":
                    raise RuntimeError(f"Unexpected global pooling at layer {index}")
                flattened = previous.flatten(2)
                value = (
                    torch.logsumexp(5 * flattened, dim=2) - math.log(flattened.shape[2])
                ) / 5
            else:
                raise RuntimeError(f"Unsupported legacy layer {kind!r} at index {index}")
        values.append(value)
    return values[-1]


def to_float32_audio(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.signedinteger):
        return audio.astype(np.float32) / float(2 ** (np.iinfo(audio.dtype).bits - 1))
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    return audio.astype(np.float32, copy=False)


def fixed_window(audio: np.ndarray, width: int, crop_index: int, crops_per_record: int) -> np.ndarray:
    """Take evenly distributed, deterministic 3-s windows from a recording."""
    if crops_per_record < 1 or not 0 <= crop_index < crops_per_record:
        raise ValueError("invalid crop index")
    if len(audio) < width:
        return np.pad(audio, (0, width - len(audio)))
    spare = len(audio) - width
    # A single crop uses the centre; multiple crops cover different positions
    # reproducibly without adapting crop selection to labels or predictions.
    fraction = (crop_index + 0.5) / crops_per_record
    start = int(round(spare * fraction))
    return audio[start : start + width]


def make_spec(
    record: Record,
    snapshot: dict[str, Any],
    crop_index: int,
    crops_per_record: int,
    demo_root: Path,
) -> np.ndarray:
    add_demo_root(demo_root)
    from birdnet_demo import make_spectrogram

    rate, audio = wavfile.read(record.path)
    audio = to_float32_audio(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise RuntimeError(f"Expected mono/stereo WAV, got {audio.shape} for {record.path}")
    target_rate = int(snapshot["sample_rate"])
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        audio = resample_poly(audio, target_rate // divisor, rate // divisor).astype(np.float32)
    width = int(float(snapshot["spec_length"]) * target_rate)
    segment = fixed_window(audio, width, crop_index, crops_per_record)
    spec = make_spectrogram(segment, snapshot)
    if spec.shape != (64, 384):
        raise RuntimeError(f"Unexpected spectrogram shape {spec.shape} for {record.path}")
    return spec


def build_feature_cache(args: argparse.Namespace) -> None:
    split_manifest = args.run_dir / "split_manifest.csv"
    all_records = load_records(split_manifest, args.audio_root)
    class_labels = [
        next(record.label for record in all_records if record.species_index == index)
        for index in range(20)
    ]
    records = all_records
    if args.limit is not None:
        records = records[: args.limit]
        if not records:
            raise RuntimeError("--limit selected no records")
    device = resolve_device(args.device)
    model, snapshot = load_frozen_birdnet(args.demo_root, args.snapshot)
    model.to(device)
    torch.set_num_threads(args.threads)
    # Layer 153 is the frozen activation immediately before the 987-class
    # convolution at layer 154. It has 1,024 channels.
    feature_layer_index = 153
    all_features: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_splits: list[str] = []
    all_rows: list[int] = []
    all_crop_indices: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch_records = records[start : start + args.batch_size]
            batch_specs: list[np.ndarray] = []
            for record in batch_records:
                for crop_index in range(args.crops_per_record):
                    batch_specs.append(
                        make_spec(
                            record,
                            snapshot,
                            crop_index,
                            args.crops_per_record,
                            args.demo_root,
                        )
                    )
                    all_labels.append(record.species_index)
                    all_splits.append(record.split)
                    all_rows.append(record.hf_row_index)
                    all_crop_indices.append(crop_index)
            batch = torch.from_numpy(np.stack(batch_specs)[:, None]).to(device)
            features = forward_until(model, batch, feature_layer_index).cpu()
            all_features.append(features)
            completed = min(start + len(batch_records), len(records))
            print(f"cached frozen features for {completed}/{len(records)} recordings", flush=True)
    features = torch.cat(all_features, dim=0).contiguous()
    if features.ndim != 4 or features.shape[1] != 1024:
        raise RuntimeError(f"Unexpected frozen feature shape {tuple(features.shape)}")
    cache = {
        "format_version": CACHE_FORMAT_VERSION,
        "feature_layer_index": feature_layer_index,
        "features": features,
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "splits": all_splits,
        "hf_row_indices": all_rows,
        "crop_indices": all_crop_indices,
        "crops_per_record": args.crops_per_record,
        "class_labels": class_labels,
        "snapshot_path": str(args.snapshot),
        "snapshot_sha256": sha256(args.snapshot),
        "original_class_count": len(model.classes),
        "expanded_class_count": len(model.classes) + 20,
    }
    atomic_torch_save(args.run_dir / "features.pt", cache)
    report = {
        "cache": str(args.run_dir / "features.pt"),
        "feature_shape": list(features.shape),
        "records": len(records),
        "examples": len(all_labels),
        "crops_per_record": args.crops_per_record,
        "split_examples": dict(Counter(all_splits)),
        "source_snapshot_sha256": cache["snapshot_sha256"],
        "device": str(device),
    }
    atomic_json_write(args.run_dir / "cache_report.json", report)
    print(json.dumps(report, indent=2))


def load_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("format_version") != CACHE_FORMAT_VERSION:
        raise RuntimeError("Unrecognized cache format")
    if cache.get("original_class_count") != 987 or cache.get("expanded_class_count") != 1007:
        raise RuntimeError("Cache does not represent a 987 + 20 classifier")
    features = cache["features"]
    labels = cache["labels"]
    if features.ndim != 4 or features.shape[1] != 1024 or len(features) != len(labels):
        raise RuntimeError("Malformed frozen feature cache")
    return cache


class AppendedHead(nn.Module):
    """Twenty trainable class filters plus immutable copies of the old head."""

    def __init__(self, frozen_model: Any, new_class_count: int = 20, dropout: float = 0.5) -> None:
        super().__init__()
        old_weight = frozen_model.n154_W.detach().clone()
        old_bias = frozen_model.n154_b.detach().clone()
        if tuple(old_weight.shape) != (987, 1024, 1, 1) or tuple(old_bias.shape) != (987,):
            raise RuntimeError(
                f"Unexpected original class head shapes {tuple(old_weight.shape)}, {tuple(old_bias.shape)}"
            )
        self.register_buffer("old_weight", old_weight)
        self.register_buffer("old_bias", old_bias)
        # BirdNET-V1 applies dropout immediately before its original class
        # convolution. The base is frozen here, so reproduce that final regularizer
        # for the appended head only.
        self.dropout = nn.Dropout(p=dropout)
        self.new_head = nn.Conv2d(1024, new_class_count, kernel_size=1, bias=True)

    @staticmethod
    def pool_logits(logits: torch.Tensor) -> torch.Tensor:
        flattened = logits.flatten(2)
        return (torch.logsumexp(5 * flattened, dim=2) - math.log(flattened.shape[2])) / 5

    def new_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.pool_logits(self.new_head(self.dropout(features)))

    def new_scores(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.new_logits(features))

    def all_scores(self, features: torch.Tensor) -> torch.Tensor:
        old_logits = self.pool_logits(F.conv2d(features, self.old_weight, self.old_bias))
        return torch.cat((torch.sigmoid(old_logits), self.new_scores(features)), dim=1)


def split_indices(splits: Iterable[str], wanted: str) -> torch.Tensor:
    return torch.tensor([index for index, split in enumerate(splits) if split == wanted], dtype=torch.long)


def make_targets(labels: torch.Tensor, classes: int = 20) -> torch.Tensor:
    return F.one_hot(labels, num_classes=classes).to(dtype=torch.float32)


def prediction_metrics(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, Any]:
    targets = make_targets(labels).to(dtype=torch.bool)
    predictions = scores >= threshold
    true_positive = (predictions & targets).sum(dim=0).to(dtype=torch.float64)
    false_positive = (predictions & ~targets).sum(dim=0).to(dtype=torch.float64)
    false_negative = (~predictions & targets).sum(dim=0).to(dtype=torch.float64)
    precision = torch.where(true_positive + false_positive > 0, true_positive / (true_positive + false_positive), 0.0)
    recall = torch.where(true_positive + false_negative > 0, true_positive / (true_positive + false_negative), 0.0)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    totals = {
        "true_positive": int(true_positive.sum().item()),
        "false_positive": int(false_positive.sum().item()),
        "false_negative": int(false_negative.sum().item()),
    }
    micro_precision = totals["true_positive"] / max(1, totals["true_positive"] + totals["false_positive"])
    micro_recall = totals["true_positive"] / max(1, totals["true_positive"] + totals["false_negative"])
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "threshold": threshold,
        "top1_accuracy": float((scores.argmax(dim=1) == labels).float().mean().item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "events": int(predictions.sum().item()),
        "per_class": [
            {
                "class_index": class_index,
                "precision": float(precision[class_index].item()),
                "recall": float(recall[class_index].item()),
                "f1": float(f1[class_index].item()),
                "true_positive": int(true_positive[class_index].item()),
                "false_positive": int(false_positive[class_index].item()),
                "false_negative": int(false_negative[class_index].item()),
            }
            for class_index in range(len(precision))
        ],
    }


def infer_new_scores(head: AppendedHead, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            outputs.append(head.new_scores(features[start : start + batch_size]).cpu())
    return torch.cat(outputs, dim=0)


def infer_new_logits(head: AppendedHead, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            outputs.append(head.new_logits(features[start : start + batch_size]).cpu())
    return torch.cat(outputs, dim=0)


def select_threshold(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, list[dict[str, Any]]]:
    candidates = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    sweep = [prediction_metrics(scores, labels, threshold) for threshold in candidates]
    # Prefer a higher threshold when F1 ties; it is less permissive for the
    # unlabeled Denoising visualization, while the validation rule stays fixed.
    best = max(sweep, key=lambda row: (row["macro_f1"], row["threshold"]))
    return float(best["threshold"]), sweep


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    cache = load_cache(args.run_dir / "features.pt")
    if cache["snapshot_sha256"] != sha256(args.snapshot):
        raise RuntimeError("Snapshot checksum differs from the feature-cache source")
    frozen_model, _ = load_frozen_birdnet(args.demo_root, args.snapshot)
    head = AppendedHead(frozen_model).to(device)
    features = cache["features"].float().to(device)
    labels = cache["labels"].long().to(device)
    train_index = split_indices(cache["splits"], "train")
    validation_index = split_indices(cache["splits"], "validation")
    test_index = split_indices(cache["splits"], "test")
    if min(len(train_index), len(validation_index), len(test_index)) == 0:
        raise RuntimeError("Feature cache does not contain all three data splits")
    optimizer = torch.optim.Adam(head.new_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_bce = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    generator = torch.Generator().manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = train_index[torch.randperm(len(train_index), generator=generator)]
        total_loss = 0.0
        total_examples = 0
        for start in range(0, len(permutation), args.batch_size):
            batch_index = permutation[start : start + args.batch_size]
            device_batch_index = batch_index.to(device)
            logits = head.new_logits(features[device_batch_index])
            loss = criterion(logits, make_targets(labels[device_batch_index]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_index)
            total_examples += len(batch_index)
        validation_logits = infer_new_logits(head, features[validation_index.to(device)], args.batch_size)
        validation_labels = labels[validation_index.to(device)].cpu()
        validation_bce = float(
            criterion(validation_logits, make_targets(validation_labels)).item()
        )
        scheduler.step(validation_bce)
        validation_scores = torch.sigmoid(validation_logits)
        validation_metrics = prediction_metrics(
            validation_scores, validation_labels, threshold=0.5
        )
        epoch_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_bce": total_loss / total_examples,
            "validation_bce": validation_bce,
            "validation_top1_accuracy": validation_metrics["top1_accuracy"],
            "validation_macro_f1_at_0_5": validation_metrics["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        if validation_bce < best_validation_bce - 1e-12:
            best_validation_bce = validation_bce
            best_state = {name: value.detach().clone() for name, value in head.new_head.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after epoch {epoch}; best epoch was {best_epoch}", flush=True)
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    head.new_head.load_state_dict(best_state)
    validation_scores = infer_new_scores(head, features[validation_index.to(device)], args.batch_size)
    validation_labels = labels[validation_index.to(device)].cpu()
    chosen_threshold, validation_sweep = select_threshold(validation_scores, validation_labels)
    validation_metrics = prediction_metrics(validation_scores, validation_labels, chosen_threshold)
    test_scores = infer_new_scores(head, features[test_index.to(device)], args.batch_size)
    test_metrics = prediction_metrics(
        test_scores, labels[test_index.to(device)].cpu(), chosen_threshold
    )
    class_labels = list(cache["class_labels"])
    for metrics in (validation_metrics, test_metrics):
        for row in metrics["per_class"]:
            row["label"] = class_labels[row["class_index"]]
    checkpoint = {
        "format_version": 1,
        "description": "20 class head appended to frozen 987-class BirdNET-V1",
        "source_snapshot": str(args.snapshot),
        "source_snapshot_sha256": cache["snapshot_sha256"],
        "original_class_count": 987,
        "new_class_count": 20,
        "expanded_class_count": 1007,
        "new_class_labels": class_labels,
        "new_head_state_dict": {name: value.cpu() for name, value in head.new_head.state_dict().items()},
        "best_epoch": best_epoch,
        "selected_threshold": chosen_threshold,
        "selection_rule": "highest validation macro F1; higher threshold breaks exact ties",
    }
    checkpoint_path = args.run_dir / "append20_head.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "status": "completed",
        "architecture": {
            "frozen_feature_layer": 153,
            "frozen_feature_channels": 1024,
            "original_output_classes": 987,
            "appended_output_classes": 20,
            "expanded_output_classes": 1007,
            "training_scope": "only the appended 20-class 1x1 convolution",
            "old_class_preservation": "the original 987 weights and all earlier layers are frozen",
            "appended_head_dropout": 0.5,
        },
        "data": {
            "cache": str(args.run_dir / "features.pt"),
            "crops_per_record": cache["crops_per_record"],
            "examples": {split: int((torch.tensor([value == split for value in cache["splits"]])).sum()) for split in SPLITS},
            "label_scope": "one recording-level iNaturalist species label; no timestamp annotations",
        },
        "optimization": {
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "best_validation_bce": best_validation_bce,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lr_schedule": {
                "name": "ReduceLROnPlateau",
                "monitor": "validation BCE",
                "factor": 0.5,
                "patience": args.lr_patience,
            },
            "patience": args.patience,
            "device": str(device),
        },
        "threshold": {
            "selected_on": "validation split only",
            "selected_threshold": chosen_threshold,
            "sweep": validation_sweep,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
        "caveat": (
            "The held-out recordings support an internal pilot evaluation of the 20 new labels. "
            "The source mirror omits original recording identifiers and these are weak recording labels; "
            "they do not establish performance on the unlabeled Denoising audio."
        ),
    }
    atomic_json_write(args.run_dir / "training_report.json", report)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "threshold": chosen_threshold,
        "validation_bce": best_validation_bce,
        "validation_top1": validation_metrics["top1_accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_top1": test_metrics["top1_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)

    def shared(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
        command.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
        command.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)

    prepare_parser = commands.add_parser("prepare", help="make a reproducible grouped pilot split")
    shared(prepare_parser)
    prepare_parser.add_argument("--seed", type=int, default=365)
    prepare_parser.add_argument(
        "--split-counts", type=parse_split_counts, default=parse_split_counts("train=35,validation=7,test=8")
    )
    prepare_parser.set_defaults(func=prepare_split)

    cache_parser = commands.add_parser("cache", help="cache frozen 1,024-channel features for all pilot WAVs")
    shared(cache_parser)
    cache_parser.add_argument("--batch-size", type=int, default=4)
    cache_parser.add_argument("--threads", type=int, default=4)
    cache_parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    cache_parser.add_argument("--crops-per-record", type=int, default=1)
    cache_parser.add_argument("--limit", type=int, help="smoke-test only: cache the first N split rows")
    cache_parser.set_defaults(func=build_feature_cache)

    train_parser = commands.add_parser("train", help="train and evaluate the appended 20-class head")
    shared(train_parser)
    train_parser.add_argument("--seed", type=int, default=365)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--patience", type=int, default=12)
    train_parser.add_argument("--lr-patience", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--threads", type=int, default=4)
    train_parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.set_defaults(func=train)
    return top


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
