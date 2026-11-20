# Ecology Reporting — Landing Page

A Flask + React starter for the ecology reporting site, modeled on the
jungle-themed mockup: canopy background, mascot with an instructions
callout, and a live projects panel backed by a small API.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Structure

- `app.py` — Flask app. Serves the page and a small `/api/projects` API
  (GET / POST / DELETE), currently backed by an in-memory list — swap
  in a real database when you're ready.
- `templates/index.html` — HTML shell that loads fonts and mounts React.
- `static/app.jsx` — the React UI (hero, SVG canopy/mascot, instructions
  modal, projects panel). Uses Babel-in-browser, so there's no build
  step; for production, compile it with a real bundler (Vite/CRA) and
  swap the CDN `<script>` tags for the built bundle.
- `static/styles.css` — the design system (colors, type, layout).

## Notes

- The mascot is a tree frog rather than the monkey in the original
  mock — frogs are a common ecological-health indicator species, which
  felt more purposeful for a monitoring tool. Swap `MascotFrog` in
  `app.jsx` for another creature if you'd rather keep the monkey.
- "New project" and the trash icon on each row are wired to the API
  already; deleting removes it from the in-memory store (refresh
  resets to the two seed projects).
