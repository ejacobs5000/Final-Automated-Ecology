"""
Species-detection aggregation for Ecology Reporting projects.

Detections always come from a real analysis pipeline — the delivered
BirdNET pilot CSV (birdnet_results.py) or live single-file inference
(birdnet_inference.py). This module holds no synthetic/fake data; it just
rolls up whatever real detections a project has into every chart/table the
project dashboard needs, and tracks the species catalog those pipelines
register via register_species().
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta

SPECIES_BY_KEY = {}

NOTABLE_STATUSES = {"Near Threatened", "Vulnerable", "Endangered", "Critically Endangered"}


def register_species(entries: list[dict]) -> None:
    """Merge additional species-catalog entries (e.g. from a real analysis
    pipeline) in, so build_dashboard() can resolve their detections too."""
    for entry in entries:
        SPECIES_BY_KEY.setdefault(entry["key"], entry)

MAX_GENERATED_DAYS = 90
MAX_HOUR_BINS = 480

DATA_DISPLAY_KEYS = [
    "ts_per_species",
    "diel_per_species",
    "calendar_heatmap",
    "taxonomic_breakdown",
    "species_leaderboard",
    "species_accumulation",
    "new_species_flagged",
    "notable_rare",
    "species_lookup",
    "comparison_timeseries",
    "comparison_diel",
]


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        start, end = end, start
    total = (end - start).days + 1
    if total <= MAX_GENERATED_DAYS:
        return [start + timedelta(days=i) for i in range(total)]
    stride = total / MAX_GENERATED_DAYS
    return sorted({start + timedelta(days=int(i * stride)) for i in range(MAX_GENERATED_DAYS)})


def _hour_range(start: date, end: date) -> list[datetime]:
    """Hourly timeline across the project's date range, for a true time-of-
    day timeline rather than one point per calendar day — a single-day
    project still gets 24 meaningful points instead of one flat one."""
    first = datetime(start.year, start.month, start.day)
    last = datetime(end.year, end.month, end.day, 23)
    total_hours = int((last - first).total_seconds() // 3600) + 1
    if total_hours <= MAX_HOUR_BINS:
        return [first + timedelta(hours=i) for i in range(total_hours)]
    stride = total_hours / MAX_HOUR_BINS
    return sorted({first + timedelta(hours=int(i * stride)) for i in range(MAX_HOUR_BINS)})


_PALETTE = [
    "#3f8862", "#e7a94c", "#6b4423", "#8fcb7c", "#4a90a4",
    "#b2574a", "#9b7fc7", "#2f6a4a", "#d98f3a", "#5c9ea8",
]


def _color_for(index: int) -> str:
    return _PALETTE[index % len(_PALETTE)]


def build_dashboard(project: dict) -> dict:
    detections = project["detections"]
    sites = project["sites"]
    start, end = _parse_date(project["start_date"]), _parse_date(project["end_date"])
    days = _date_range(start, end)
    day_strs = [d.isoformat() for d in days]
    hours_timeline = _hour_range(start, end)
    hour_bin_strs = [h.strftime("%Y-%m-%dT%H") for h in hours_timeline]

    counts_by_species = defaultdict(int)
    by_species_by_hourbin = defaultdict(lambda: defaultdict(int))
    by_species_by_hour = defaultdict(lambda: [0] * 24)
    by_date_total = defaultdict(int)
    by_site_by_hourbin = defaultdict(lambda: defaultdict(int))
    by_site_by_hour = defaultdict(lambda: [0] * 24)
    first_seen = {}

    for d in detections:
        species_key = d["species_key"]
        ts = datetime.fromisoformat(d["timestamp"])
        date_str = ts.date().isoformat()
        hour_bin = ts.strftime("%Y-%m-%dT%H")
        counts_by_species[species_key] += 1
        by_species_by_hourbin[species_key][hour_bin] += 1
        by_species_by_hour[species_key][ts.hour] += 1
        by_date_total[date_str] += 1
        by_site_by_hourbin[d["site"]][hour_bin] += 1
        by_site_by_hour[d["site"]][ts.hour] += 1
        if species_key not in first_seen or date_str < first_seen[species_key]:
            first_seen[species_key] = date_str

    species_present = sorted(counts_by_species.keys(), key=lambda k: -counts_by_species[k])

    def label(key):
        return SPECIES_BY_KEY[key]["common_name"]

    top_species = species_present[:8]
    ts_per_species = {
        "hours": hour_bin_strs,
        "series": [
            {
                "key": key,
                "label": label(key),
                "color": _color_for(i),
                "data": [by_species_by_hourbin[key].get(h, 0) for h in hour_bin_strs],
            }
            for i, key in enumerate(top_species)
        ],
    }

    diel_per_species = {
        "hours": list(range(24)),
        "series": [
            {
                "key": key,
                "label": label(key),
                "color": _color_for(i),
                "data": by_species_by_hour[key],
            }
            for i, key in enumerate(top_species)
        ],
    }

    calendar_heatmap = [{"date": d, "count": by_date_total.get(d, 0)} for d in day_strs]

    # Grouped by taxonomic family rather than class — nearly every species
    # this app ever sees is a bird (BirdNET only detects birds), so class
    # would always collapse to one useless "Bird" bucket. Family actually
    # splits them into meaningful groups (parrots, owls, raptors, ...).
    taxonomy = defaultdict(list)
    for key in species_present:
        s = SPECIES_BY_KEY[key]
        family = s.get("family_name") or "Unclassified"
        taxonomy[family].append({"key": key, "label": s["common_name"], "count": counts_by_species[key]})
    taxonomic_breakdown = [
        {"family_name": fam, "species": sorted(items, key=lambda x: -x["count"])}
        for fam, items in sorted(taxonomy.items())
    ]

    leaderboard = [
        {"rank": i + 1, "key": key, "label": label(key), "count": counts_by_species[key]}
        for i, key in enumerate(species_present[:10])
    ]

    seen_so_far = set()
    accumulation = []
    cumulative_by_date = defaultdict(set)
    for d in detections:
        cumulative_by_date[datetime.fromisoformat(d["timestamp"]).date().isoformat()].add(d["species_key"])
    for d in day_strs:
        seen_so_far |= cumulative_by_date.get(d, set())
        accumulation.append({"date": d, "cumulative": len(seen_so_far)})

    new_species_flagged = []
    if len(day_strs) >= 4:
        cutoff = day_strs[int(len(day_strs) * 0.75)]
        for key, seen_date in first_seen.items():
            if seen_date >= cutoff:
                new_species_flagged.append({"key": key, "label": label(key), "first_seen": seen_date})
        new_species_flagged.sort(key=lambda x: x["first_seen"])

    notable_rare = [
        {
            "key": key,
            "label": label(key),
            "status": SPECIES_BY_KEY[key]["status"],
            "blurb": SPECIES_BY_KEY[key]["blurb"],
            "count": counts_by_species[key],
        }
        for key in species_present
        if SPECIES_BY_KEY[key]["status"] in NOTABLE_STATUSES
    ]

    species_lookup = [
        {
            "key": key,
            "common_name": SPECIES_BY_KEY[key]["common_name"],
            "scientific_name": SPECIES_BY_KEY[key]["scientific_name"],
            "class_name": SPECIES_BY_KEY[key]["class_name"],
            "status": SPECIES_BY_KEY[key]["status"],
            "blurb": SPECIES_BY_KEY[key]["blurb"],
        }
        for key in species_present
    ]

    by_source_file = defaultdict(list)
    for d in detections:
        if d.get("source_file"):
            by_source_file[d["source_file"]].append(d)

    per_audio = []
    for meta in project.get("audio_meta") or []:
        file_detections = sorted(by_source_file.get(meta["filename"], []), key=lambda d: d["timestamp"])
        per_audio.append({
            "filename": meta["filename"],
            "site": meta["site"],
            "date": meta["date"],
            "time": meta["time"],
            "detections": [
                {
                    "id": d["id"],
                    "species_key": d["species_key"],
                    "label": label(d["species_key"]),
                    "timestamp": d["timestamp"],
                    "confidence": d.get("confidence"),
                    "user_overridden": d.get("user_overridden", False),
                }
                for d in file_detections
            ],
        })

    comparisons = None
    if project.get("multiple_sites") and len(sites) > 1:
        comparisons = {
            "timeseries": {
                "hours": hour_bin_strs,
                "series": [
                    {
                        "key": site,
                        "label": site,
                        "color": _color_for(i),
                        "data": [by_site_by_hourbin[site].get(h, 0) for h in hour_bin_strs],
                    }
                    for i, site in enumerate(sites)
                ],
            },
            "diel": {
                "hours": list(range(24)),
                "series": [
                    {"key": site, "label": site, "color": _color_for(i), "data": by_site_by_hour[site]}
                    for i, site in enumerate(sites)
                ],
            },
        }

    return {
        "detection_count": len(detections),
        "species_richness": len(species_present),
        "ts_per_species": ts_per_species,
        "diel_per_species": diel_per_species,
        "calendar_heatmap": calendar_heatmap,
        "taxonomic_breakdown": taxonomic_breakdown,
        "leaderboard": leaderboard,
        "species_accumulation": accumulation,
        "new_species_flagged": new_species_flagged,
        "notable_rare": notable_rare,
        "species_lookup": species_lookup,
        "comparisons": comparisons,
        "per_audio": per_audio,
    }
