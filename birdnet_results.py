"""
Adapter: real BirdNET-pilot candidate detections -> this app's dashboard shapes.

BirdNet_result/ (pulled from the team's denoising/analysis handoff) ships a
fixed, pre-computed delivery for its original 30-recording demo campaign —
see BirdNet_result/docs/data_contract.md. This module reads that delivered
CSV once at startup and reshapes it into the same {species_key, site,
timestamp, confidence} detection rows — and a matching species-catalog
fragment — that ecology_data.build_dashboard() already knows how to
aggregate, so the original delivery flows through the same charts as any
other project.

For a freshly uploaded recording (not part of that fixed campaign), see
birdnet_inference.py instead — it runs the real model live, reusing
catalog_entry()/species_slug() from this module so both paths key the same
species consistently.

Per the handoff's interpretation rules: these are uncalibrated candidate
detections, not confirmed sightings or probabilities — kept out of
conclusions, just carried through as event counts (which is the contract's
own suggested use of candidate_detections.csv).
"""
from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "BirdNet_result" / "results"
CANDIDATE_DETECTIONS_CSV = RESULTS_DIR / "candidate_detections.csv"

SITE_LABEL = "BirdNET pilot recordings"

_FILENAME_RE = re.compile(r"^(\d{8})_(\d{6})_(\w+)_v\d+\.\w+$")


def species_slug(scientific_name: str) -> str:
    return "bn_" + re.sub(r"[^a-z0-9]+", "_", scientific_name.lower()).strip("_")


def catalog_entry(scientific_name: str, common_name: str) -> dict:
    """A species-catalog entry shaped for ecology_data.register_species().

    Shared by the pre-computed CSV loader here and by live single-file
    inference (birdnet_inference.py), so a species gets the same key and
    the same honest "Not assessed" status regardless of which path produced
    the detection.
    """
    return {
        "key": species_slug(scientific_name),
        "common_name": common_name,
        "scientific_name": scientific_name,
        "class_name": "Bird",
        "family_name": None,
        "status": "Not assessed",
        "commonness": "moderate",
        "windows": [(0, 23)],
        "blurb": "BirdNET pilot-model candidate detection; not a confirmed sighting.",
    }


def _file_timestamp(filename: str) -> datetime | None:
    match = _FILENAME_RE.match(filename)
    if not match:
        return None
    date_part, time_part, _recorder_id = match.groups()
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")


def available() -> bool:
    return CANDIDATE_DETECTIONS_CSV.exists()


def load_species_catalog() -> list[dict]:
    """Species-catalog fragment for every species named in the candidate CSV.

    Conservation status isn't part of the handoff's data contract, so it's
    left as "Not assessed" rather than guessed — the notable/rare callout
    only flags species status is actually known to be of elevated concern.
    """
    if not available():
        return []

    seen = {}
    with open(CANDIDATE_DETECTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entry = catalog_entry(row["scientific_name"], row["common_name"])
            seen.setdefault(entry["key"], entry)
    return list(seen.values())


def load_detections() -> list[dict]:
    """Candidate-detection rows reshaped into this app's detection schema."""
    if not available():
        return []

    detections = []
    with open(CANDIDATE_DETECTIONS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            base = _file_timestamp(row["filename"])
            if base is None:
                continue
            timestamp = base + timedelta(seconds=float(row["start_s"]))
            detections.append({
                "species_key": species_slug(row["scientific_name"]),
                "site": SITE_LABEL,
                "timestamp": timestamp.isoformat(),
                "confidence": round(float(row["score"]), 4),
                "source_file": row["filename"],
            })
    detections.sort(key=lambda d: d["timestamp"])
    return detections


def list_source_files() -> list[str]:
    all_scores_dir = RESULTS_DIR / "all_scores"
    if not all_scores_dir.exists():
        return []
    return sorted(p.name.replace("_all_scores.npz", ".flac") for p in all_scores_dir.glob("*_all_scores.npz"))


def audio_meta() -> list[dict]:
    """Per-recording {filename, site, date, time} for the "per audio" report
    section — real date/time parsed from each delivered file's own name."""
    entries = []
    for filename in list_source_files():
        ts = _file_timestamp(filename)
        entries.append({
            "filename": filename,
            "site": SITE_LABEL,
            "date": ts.date().isoformat() if ts else "",
            "time": ts.strftime("%H:%M") if ts else "",
        })
    return entries


def date_range() -> tuple[str, str] | None:
    detections = load_detections()
    if not detections:
        return None
    dates = sorted({d["timestamp"][:10] for d in detections})
    return dates[0], dates[-1]
