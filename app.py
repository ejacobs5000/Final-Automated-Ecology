"""
Ecology Reporting — backend
A small Flask API that serves the React landing page, the per-project
dashboard page, and a simple in-memory store of report projects. Swap the
in-memory list for a real database when you're ready to persist data.

Uploaded audio is run through the real BirdNET pilot model
(birdnet_inference.py) when BirdNet_result/artifacts/BirdNET_Soundscape_Model.pkl
is present. Without that file (it's a licensed ~300MB download, not
committed to git — see BirdNet_result/artifacts/MODEL_SETUP.md), a project
with audio just has no detections, with analysis_errors explaining why.
There is no synthetic/fake fallback — a project's dashboard only ever
reflects real model output, never invented data.

Whenever a species is newly registered (from either source), its real
conservation status and taxonomic family are looked up via
species_enrichment.py (IUCN Red List status and family, both via
Wikidata — no API key needed). A species stays "Not assessed" (status)
or falls back to "Unclassified" (family) only when that lookup finds
nothing.

One project (sample_data.py) is a fixed, hand-built demo so there's
something to show the full report layout on — it is never used as a
fallback for a project whose real analysis fails or is unavailable.
"""
import string
import tempfile
from datetime import date, datetime
from pathlib import Path
import itertools

from flask import Flask, jsonify, request, render_template, abort

import birdnet_inference
import birdnet_results
import species_enrichment
import sample_data
from ecology_data import DATA_DISPLAY_KEYS, SPECIES_BY_KEY, build_dashboard, register_species

app = Flask(__name__)

_id_counter = itertools.count(3)
_detection_id_counter = itertools.count(1)


def _stamp_ids(detections: list[dict]) -> list[dict]:
    """Every detection needs a stable id (for the manual-override/reject
    endpoints) and a source_file slot (for the per-audio breakdown)."""
    for d in detections:
        d.setdefault("id", next(_detection_id_counter))
        d.setdefault("source_file", None)
    return detections


def _site_names(multiple_sites: bool, count: int) -> list[str]:
    if not multiple_sites:
        return ["Main Site"]
    count = max(2, min(count, len(string.ascii_uppercase)))
    return [f"Site {string.ascii_uppercase[i]}" for i in range(count)]


def _basic_audio_meta(audio_filenames, sites, start_date) -> list[dict]:
    """Fallback per-file metadata for projects that predate per-file date/
    time/site entry (the seed demo projects) — round-robins sites and
    anchors every file at the project's start date, since there's no real
    per-recording metadata to draw on for those."""
    return [
        {"filename": f, "site": sites[i % len(sites)], "date": start_date, "time": "00:00"}
        for i, f in enumerate(audio_filenames)
    ]


def _seed_project(project_id, name, status, multiple_sites, sites, start_date, end_date,
                   audio_filenames, data_displays=None, detections=None, analysis_errors=None,
                   audio_meta=None, analysis_source=None):
    project = {
        "id": project_id,
        "name": name,
        "status": status,
        "updated": "2 days ago" if status != "Draft" else "1 week ago",
        "multiple_sites": multiple_sites,
        "sites": sites,
        "start_date": start_date,
        "end_date": end_date,
        "audio_filenames": audio_filenames,
        "audio_meta": audio_meta or _basic_audio_meta(audio_filenames, sites, start_date),
        "data_displays": data_displays or list(DATA_DISPLAY_KEYS),
        "detections": [],
        "analysis_source": "none",
        "analysis_errors": analysis_errors or [],
    }
    if detections is not None:
        project["detections"] = detections
        project["analysis_source"] = "real"
    if analysis_source is not None:
        project["analysis_source"] = analysis_source
    _stamp_ids(project["detections"])
    return project


def _analyze_uploads(files, audio_meta):
    """Run real BirdNET inference on each uploaded file using its own
    user-entered date/time/site (audio_meta[i] matches files[i]). Returns
    (detections, succeeded, errors) — succeeded is False only if every file
    failed to process (too short, corrupt, unsupported format, ...); a real
    run that legitimately finds nothing is not a failure. errors carries a
    human-readable reason per failed file, so a failure is never silent."""
    detections = []
    succeeded = False
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, file in enumerate(files):
            meta = audio_meta[index]
            safe_name = Path(file.filename).name or f"upload_{index}.wav"
            upload_path = Path(tmp) / f"{index}_{safe_name}"
            file.save(upload_path)
            base = datetime.strptime(f"{meta['date']} {meta['time']}", "%Y-%m-%d %H:%M")
            try:
                file_detections, catalog_entries = birdnet_inference.analyze_audio_file(upload_path, base)
            except Exception as exc:  # unsupported/corrupt/too-short audio for this one file
                message = f"{file.filename}: {exc}"
                print(f"BirdNET inference failed for {message}", flush=True)
                errors.append(message)
                continue
            succeeded = True
            species_enrichment.enrich_species(catalog_entries)
            register_species(catalog_entries)
            for d in file_detections:
                d["site"] = meta["site"]
                d["source_file"] = meta["filename"]
            detections.extend(file_detections)
    return detections, succeeded, errors


def _birdnet_seed_project(project_id):
    """Real candidate detections from the team's BirdNET pilot handoff
    (BirdNet_result/), run through the same dashboard as the demo projects."""
    pilot_catalog = birdnet_results.load_species_catalog()
    species_enrichment.enrich_species(pilot_catalog)
    register_species(pilot_catalog)
    detections = birdnet_results.load_detections()
    bounds = birdnet_results.date_range() or (date.today().isoformat(), date.today().isoformat())
    non_comparison_displays = [k for k in DATA_DISPLAY_KEYS if not k.startswith("comparison_")]
    return _seed_project(
        project_id, "BirdNET Pilot — Candidate Detections", "In progress",
        multiple_sites=False, sites=[birdnet_results.SITE_LABEL],
        start_date=bounds[0], end_date=bounds[1],
        audio_filenames=birdnet_results.list_source_files(),
        audio_meta=birdnet_results.audio_meta(),
        data_displays=non_comparison_displays,
        detections=detections,
    )


def _sample_seed_project(project_id):
    """ONE fixed, hand-specified preview project (sample_data.py) so
    there's a fully-populated example report to look at. Not a fallback:
    nothing here is wired into project creation or upload — see
    sample_data.py. analysis_source="sample" is only an internal
    bookkeeping marker (so this one fixture is easy to find in code); the
    frontend renders it exactly like a normal project, no special badge."""
    catalog = sample_data.sample_species_catalog()
    species_enrichment.enrich_species(catalog)
    register_species(catalog)
    return _seed_project(
        project_id, sample_data.PROJECT_NAME, "In progress",
        multiple_sites=True, sites=sample_data.SITES,
        start_date=sample_data.START_DATE, end_date=sample_data.END_DATE,
        audio_filenames=[],
        analysis_source="sample",
        detections=sample_data.sample_detections(),
    )


_today = date.today()
projects = [
    _seed_project(
        2, "Canopy Bird Count — Q3", "Draft",
        multiple_sites=False, sites=["Canopy Plot 1"],
        start_date=_today.isoformat(), end_date=_today.isoformat(),
        audio_filenames=[],
    ),
]

if birdnet_results.available():
    projects.append(_birdnet_seed_project(next(_id_counter)))

projects.append(_sample_seed_project(next(_id_counter)))


def _find_project(project_id):
    return next((p for p in projects if p["id"] == project_id), None)


def _project_summary(project):
    return {
        "id": project["id"],
        "name": project["name"],
        "status": project["status"],
        "updated": project["updated"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/projects/<int:project_id>")
def project_page(project_id):
    if _find_project(project_id) is None:
        abort(404)
    return render_template("project.html")


@app.route("/api/projects", methods=["GET"])
def get_projects():
    return jsonify([_project_summary(p) for p in projects])


@app.route("/api/projects", methods=["POST"])
def create_project():
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Project name is required."}), 400

    multiple_sites = request.form.get("multiple_sites") == "true"
    site_names_input = [s.strip() for s in request.form.getlist("site_names") if s.strip()]
    start_date = request.form.get("start_date") or date.today().isoformat()
    end_date = request.form.get("end_date") or start_date
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    data_displays = request.form.getlist("data_displays") or list(DATA_DISPLAY_KEYS)
    files = [f for f in request.files.getlist("audio") if f.filename]
    audio_filenames = [f.filename for f in files]

    project_id = next(_id_counter)
    if multiple_sites:
        sites = site_names_input if len(site_names_input) >= 2 else _site_names(True, 2)
    else:
        sites = ["Main Site"]
    status = "In progress" if audio_filenames else "Draft"

    # Per-file date/time/site, one entry per uploaded file in the same order
    # (so the same bird heard at the same moment across sites, or across
    # different times at one site, is recorded as what it actually is).
    audio_dates = request.form.getlist("audio_date")
    audio_times = request.form.getlist("audio_time")
    audio_sites = request.form.getlist("audio_site")
    audio_meta = [
        {
            "filename": f.filename,
            "date": audio_dates[i] if i < len(audio_dates) and audio_dates[i] else start_date,
            "time": audio_times[i] if i < len(audio_times) and audio_times[i] else "00:00",
            "site": audio_sites[i] if i < len(audio_sites) and audio_sites[i] else sites[0],
        }
        for i, f in enumerate(files)
    ]

    detections = None
    analysis_errors = []
    if files and birdnet_inference.model_available():
        real_detections, succeeded, analysis_errors = _analyze_uploads(files, audio_meta)
        if succeeded:
            detections = real_detections
    elif files:
        analysis_errors = ["No BirdNET model installed on this server."]

    project = _seed_project(
        project_id, name, status,
        multiple_sites=multiple_sites, sites=sites,
        start_date=start_date, end_date=end_date,
        analysis_errors=analysis_errors,
        audio_filenames=audio_filenames, audio_meta=audio_meta, data_displays=data_displays,
        detections=detections,
    )
    project["updated"] = "just now"  # a freshly created project, not one of the fixed seed demos
    projects.append(project)
    return jsonify(_project_summary(project)), 201


@app.route("/api/projects/<int:project_id>", methods=["GET"])
def get_project(project_id):
    project = _find_project(project_id)
    if project is None:
        return jsonify({"error": "Project not found."}), 404
    return jsonify({
        **_project_summary(project),
        "multiple_sites": project["multiple_sites"],
        "sites": project["sites"],
        "start_date": project["start_date"],
        "end_date": project["end_date"],
        "audio_filenames": project["audio_filenames"],
        "data_displays": project["data_displays"],
        "analysis_source": project["analysis_source"],
        "analysis_errors": project["analysis_errors"],
        "dashboard": build_dashboard(project),
    })


@app.route("/api/projects/<int:project_id>/audio", methods=["POST"])
def add_audio(project_id):
    """Add more recordings to an existing project after the fact — same
    per-file date/time/site metadata as project creation, appended to what's
    already there rather than starting a new project."""
    project = _find_project(project_id)
    if project is None:
        return jsonify({"error": "Project not found."}), 404

    files = [f for f in request.files.getlist("audio") if f.filename]
    if not files:
        return jsonify({"error": "No audio files provided."}), 400

    audio_dates = request.form.getlist("audio_date")
    audio_times = request.form.getlist("audio_time")
    audio_sites = request.form.getlist("audio_site")
    new_audio_meta = [
        {
            "filename": f.filename,
            "date": audio_dates[i] if i < len(audio_dates) and audio_dates[i] else project["start_date"],
            "time": audio_times[i] if i < len(audio_times) and audio_times[i] else "00:00",
            "site": audio_sites[i] if i < len(audio_sites) and audio_sites[i] else project["sites"][0],
        }
        for i, f in enumerate(files)
    ]

    analysis_errors = []
    if birdnet_inference.model_available():
        real_detections, succeeded, analysis_errors = _analyze_uploads(files, new_audio_meta)
        if succeeded:
            _stamp_ids(real_detections)
            project["detections"].extend(real_detections)
            project["analysis_source"] = "real"
    else:
        analysis_errors = ["No BirdNET model installed on this server."]

    new_site_names = [m["site"] for m in new_audio_meta if m["site"] not in project["sites"]]
    if new_site_names:
        # dict.fromkeys dedupes while keeping first-seen order, in case this
        # batch itself introduces the same new site name more than once
        project["sites"].extend(dict.fromkeys(new_site_names))
        project["multiple_sites"] = True

    project["audio_filenames"].extend(f.filename for f in files)
    project["audio_meta"].extend(new_audio_meta)
    project["analysis_errors"] = analysis_errors
    project["status"] = "In progress"
    project["updated"] = "just now"

    new_dates = [m["date"] for m in new_audio_meta]
    project["start_date"] = min([project["start_date"], *new_dates])
    project["end_date"] = max([project["end_date"], *new_dates])

    return jsonify({
        **_project_summary(project),
        "dashboard": build_dashboard(project),
    }), 201


@app.route("/api/projects/<int:project_id>/audio/<path:filename>", methods=["DELETE"])
def remove_audio(project_id, filename):
    """Remove one previously-added recording and any detections tied to it —
    the reverse of add_audio, for a file a user no longer wants counted."""
    project = _find_project(project_id)
    if project is None:
        return jsonify({"error": "Project not found."}), 404
    if filename not in project["audio_filenames"]:
        return jsonify({"error": "Recording not found in this project."}), 404

    project["audio_filenames"] = [f for f in project["audio_filenames"] if f != filename]
    project["audio_meta"] = [m for m in project["audio_meta"] if m["filename"] != filename]
    project["detections"] = [d for d in project["detections"] if d.get("source_file") != filename]
    project["updated"] = "just now"
    if not project["audio_filenames"]:
        project["status"] = "Draft"
        project["analysis_source"] = "none"

    return jsonify({
        **_project_summary(project),
        "dashboard": build_dashboard(project),
    })


@app.route("/api/projects/<int:project_id>", methods=["DELETE"])
def delete_project(project_id):
    global projects
    before = len(projects)
    projects = [p for p in projects if p["id"] != project_id]
    if len(projects) == before:
        return jsonify({"error": "Project not found."}), 404
    return jsonify({"ok": True})


@app.route("/api/projects/<int:project_id>", methods=["PATCH"])
def update_project(project_id):
    """Change which report sections are displayed after the fact — the
    "Edit report" button on an already-generated report, not the
    project-creation wizard."""
    project = _find_project(project_id)
    if project is None:
        return jsonify({"error": "Project not found."}), 404
    data = request.get_json(force=True) or {}
    data_displays = data.get("data_displays")
    if not isinstance(data_displays, list) or not all(isinstance(k, str) for k in data_displays):
        return jsonify({"error": "data_displays must be a list of strings."}), 400
    project["data_displays"] = [k for k in data_displays if k in DATA_DISPLAY_KEYS]
    return jsonify({
        **_project_summary(project),
        "multiple_sites": project["multiple_sites"],
        "sites": project["sites"],
        "start_date": project["start_date"],
        "end_date": project["end_date"],
        "audio_filenames": project["audio_filenames"],
        "data_displays": project["data_displays"],
        "analysis_source": project["analysis_source"],
        "analysis_errors": project["analysis_errors"],
        "dashboard": build_dashboard(project),
    })


@app.route("/api/species", methods=["GET"])
def list_species():
    """Full known species catalog, for the manual-override dropdown on a
    detection — every species any project's data could reference."""
    return jsonify(sorted(
        [
            {"key": k, "common_name": v["common_name"], "scientific_name": v["scientific_name"]}
            for k, v in SPECIES_BY_KEY.items()
        ],
        key=lambda s: s["common_name"],
    ))


@app.route("/api/projects/<int:project_id>/detections/<int:detection_id>", methods=["PATCH"])
def override_detection(project_id, detection_id):
    """Manually correct one detection's species — a human reviewing the
    per-recording breakdown overriding what the model guessed."""
    project = _find_project(project_id)
    if project is None:
        return jsonify({"error": "Project not found."}), 404
    data = request.get_json(force=True) or {}
    species_key = data.get("species_key")
    if not species_key or species_key not in SPECIES_BY_KEY:
        return jsonify({"error": "Unknown species_key."}), 400
    detection = next((d for d in project["detections"] if d["id"] == detection_id), None)
    if detection is None:
        return jsonify({"error": "Detection not found."}), 404
    detection["species_key"] = species_key
    detection["user_overridden"] = True
    return jsonify({"ok": True})


@app.route("/api/projects/<int:project_id>/detections/<int:detection_id>", methods=["DELETE"])
def reject_detection(project_id, detection_id):
    """Remove a detection a human reviewer has decided is a false positive."""
    project = _find_project(project_id)
    if project is None:
        return jsonify({"error": "Project not found."}), 404
    before = len(project["detections"])
    project["detections"] = [d for d in project["detections"] if d["id"] != detection_id]
    if len(project["detections"]) == before:
        return jsonify({"error": "Detection not found."}), 404
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
