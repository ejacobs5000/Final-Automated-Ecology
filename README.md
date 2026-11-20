# Ecology Reporting

A Flask + React (Babel-in-browser, no build step) web app for turning
rainforest acoustic recordings into species-detection reports, powered by a
real BirdNET-based bird-classification model.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

To get real analysis on freshly uploaded audio (not just the pre-loaded
pilot project), download the BirdNET model snapshot as described in
`BirdNet_result/artifacts/MODEL_SETUP.md` — it's too large to live in this
repo. Without it, new uploads are still tracked, but detections won't run.

## Structure

- `app.py` — Flask app: project CRUD, audio upload, live BirdNET inference,
  per-detection override/reject, "Edit report" section toggles. Backed by
  an in-memory list (no real database).
- `ecology_data.py` — turns a project's detections into every report
  section (timelines, calendar heatmap, taxonomic breakdown, leaderboard,
  accumulation curve, notable/rare species, species lookup, per-recording
  breakdown, multi-site comparisons).
- `birdnet_inference.py` — runs the real BirdNET model on one freshly
  uploaded recording at a time.
- `birdnet_results.py` — adapter for the team's pre-computed 30-recording
  BirdNET pilot delivery (`BirdNet_result/`), seeded as its own project.
- `species_enrichment.py` — looks up real IUCN conservation status and
  taxonomic family from Wikidata (no synthetic/fake species data anywhere
  in the app).
- `sample_data.py` — one fixed, hand-specified rainforest-bird project so
  there's a fully-populated example report to look at; deterministic, not
  a fallback for failed analysis.
- `static/app.jsx` — landing page: jungle-themed hero with a monkey
  mascot (`MascotMonkey`), and the new-project wizard (per-file
  date/time/site metadata, audio upload).
- `static/project.jsx` — the report page: all charts/tables, "Edit
  report", add/remove audio, per-recording detection review, print/export.
- `BirdNet_result/` — the team's BirdNET pilot model, code, and delivered
  results that this app builds on.
