"""
ONE fixed preview project, for showing what a finished report looks like.

This is a hand-specified fixture, not a fallback mechanism. It is seeded
exactly once at startup as its own project and is never used as a
substitute when real BirdNET analysis is unavailable or fails —
new/existing projects never fall back to this data. There is no random
generator here; every detection below is a fixed, deterministic value, so
the same preview renders identically every run.

All six species are real rainforest birds, so their IUCN status and
taxonomic family still come from the real species_enrichment.py/Wikidata
lookup. Per-species/per-site/
per-day counts are hand-tuned to look like a plausible six-day acoustic
survey: common, widespread species (macaw, oropendola) rack up dozens of
detections with day-to-day variation, rarer species (harpy eagle,
great curassow) show up only a handful of times.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

PROJECT_NAME = "Canopy & Riverside Transect Survey — August 2026"
SITES = ["North Ridge Transect", "Riverside Transect"]
START_DATE = "2026-08-01"
END_DATE = "2026-08-06"

# (scientific_name, common_name, class_name, windows, confidence_range,
#  {site: [count_day0, count_day1, ..., count_day5]}, blurb)
_SPECIES = [
    (
        "Ara macao", "Scarlet Macaw", "Bird", [(6, 9), (16, 18)], (0.72, 0.95),
        {"North Ridge Transect": [5, 6, 4, 7, 5, 6], "Riverside Transect": [3, 4, 3, 5, 4, 4]},
        "Loud, far-carrying squawks at dawn and dusk flight times between roost and feeding trees.",
    ),
    (
        "Psarocolius montezuma", "Montezuma Oropendola", "Bird", [(6, 10)], (0.75, 0.96),
        {"North Ridge Transect": [6, 7, 5, 8, 6, 7], "Riverside Transect": [4, 5, 4, 6, 5, 5]},
        "Colonial nester; loud, liquid gurgling calls given from hanging nest colonies.",
    ),
    (
        "Lophostrix cristata", "Crested Owl", "Bird", [(19, 22)], (0.55, 0.82),
        {"North Ridge Transect": [2, 3, 2, 3, 2, 3], "Riverside Transect": [3, 4, 3, 4, 3, 3]},
        "Large nocturnal owl; deep, resonant hoots carry through the understory after dark.",
    ),
    (
        "Pharomachrus mocinno", "Resplendent Quetzal", "Bird", [(6, 8)], (0.55, 0.85),
        {"North Ridge Transect": [2, 1, 2, 0, 1, 2], "Riverside Transect": [1, 0, 1, 1, 0, 1]},
        "Cloud-forest specialist; soft, low whistled calls mostly in the first hours of daylight.",
    ),
    (
        "Crax rubra", "Great Curassow", "Bird", [(5, 6), (18, 19)], (0.5, 0.75),
        {"North Ridge Transect": [0, 0, 0, 1, 0, 0], "Riverside Transect": [1, 0, 1, 0, 1, 0]},
        "Large ground-dwelling bird; low booming calls at dawn and dusk near forest streams.",
    ),
    (
        "Harpia harpyja", "Harpy Eagle", "Bird", [(7, 10)], (0.65, 0.88),
        {"North Ridge Transect": [0, 0, 1, 0, 0, 0], "Riverside Transect": [0, 1, 0, 0, 0, 1]},
        "Apex canopy predator; detections are uncommon and worth flagging to the site lead.",
    ),
]


def _slug(scientific_name: str) -> str:
    return "demo_" + re.sub(r"[^a-z0-9]+", "_", scientific_name.lower()).strip("_")


def sample_species_catalog() -> list[dict]:
    return [
        {
            "key": _slug(scientific_name),
            "common_name": common_name,
            "scientific_name": scientific_name,
            "class_name": class_name,
            "family_name": None,
            "status": "Not assessed",
            "blurb": blurb,
        }
        for scientific_name, common_name, class_name, _windows, _conf, _counts, blurb in _SPECIES
    ]


def sample_detections() -> list[dict]:
    """Deterministic, hand-specified detection events — no randomness."""
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    detections = []

    for species_index, (scientific_name, _common, _class, windows, confidence_range, site_counts, _blurb) in enumerate(_SPECIES):
        lo, hi = confidence_range
        for site_index, site in enumerate(SITES):
            counts = site_counts[site]
            for day_index, count in enumerate(counts):
                day = start + timedelta(days=day_index)
                for event_index in range(count):
                    window = windows[(day_index + event_index) % len(windows)]
                    span = window[1] - window[0]
                    fraction = event_index / count if count > 1 else 0.5
                    hour = window[0] + int(fraction * span)
                    hour = min(hour, window[1])
                    minute = (species_index * 13 + site_index * 19 + day_index * 7 + event_index * 17) % 60
                    timestamp = day.replace(hour=hour, minute=minute)

                    step = (hi - lo) / 9 if hi > lo else 0
                    confidence_level = (species_index * 3 + site_index * 5 + day_index * 2 + event_index) % 10
                    confidence = round(lo + step * confidence_level, 2)

                    detections.append({
                        "species_key": _slug(scientific_name),
                        "site": site,
                        "timestamp": timestamp.isoformat(),
                        "confidence": confidence,
                        "source_file": None,
                    })

    detections.sort(key=lambda d: d["timestamp"])
    return detections
