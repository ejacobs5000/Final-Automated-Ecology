const { useState, useEffect, useCallback } = React;

const DATA_DISPLAY_GROUPS = [
  {
    title: "Per species",
    options: [
      { key: "ts_per_species", label: "Graph of time vs. species spotting" },
      { key: "diel_per_species", label: "Diel activity histogram (by hour of day)" },
      { key: "calendar_heatmap", label: "Calendar heatmap of activity", hint: "Only useful if data was collected across numerous days." },
    ],
  },
  {
    title: "Summative species info",
    options: [
      { key: "taxonomic_breakdown", label: "Taxonomic breakdown" },
      { key: "species_leaderboard", label: "Species leaderboard" },
      { key: "species_accumulation", label: "Species accumulation curve", hint: "Most useful across several days of data." },
      { key: "new_species_flagged", label: "New species flagged", hint: "Needs data spanning several days to tell what's new." },
      { key: "notable_rare", label: "Notable / rare species callout" },
      { key: "species_lookup", label: "Species lookup table" },
    ],
  },
  {
    title: "Comparisons",
    requiresMultipleSites: true,
    options: [
      { key: "comparison_timeseries", label: "Time vs. species spotting (multiple sites)", hint: "Only works with multiple sites." },
      { key: "comparison_diel", label: "Diel activity histogram (multiple sites)", hint: "Only works with multiple sites." },
    ],
  },
];

const ALL_DISPLAY_KEYS = DATA_DISPLAY_GROUPS.flatMap((g) => g.options.map((o) => o.key));

function CanopyTop() {
  return (
    <svg className="canopy-layer canopy-top back" viewBox="0 0 1200 160" preserveAspectRatio="none">
      <path fill="#0e2216" d="M0,0 L0,70 C60,110 120,40 180,80 C240,120 300,50 360,90
        C420,130 480,55 540,95 C600,135 660,60 720,100 C780,140 840,65 900,105
        C960,145 1020,70 1080,110 C1120,135 1160,105 1200,120 L1200,0 Z" />
    </svg>
  );
}

function CanopyTopFront() {
  return (
    <svg className="canopy-layer canopy-top" viewBox="0 0 1200 120" preserveAspectRatio="none">
      <path fill="#16311f" d="M0,0 L0,40 C50,80 100,20 150,55 C200,90 260,25 320,60
        C380,95 440,30 500,65 C560,100 620,35 680,70 C740,105 800,40 860,72
        C920,104 980,45 1040,75 C1090,100 1150,80 1200,90 L1200,0 Z" />
      <g fill="#1b3d26" opacity="0.9">
        <ellipse cx="90" cy="18" rx="34" ry="14" transform="rotate(-18 90 18)" />
        <ellipse cx="260" cy="10" rx="40" ry="16" transform="rotate(12 260 10)" />
        <ellipse cx="520" cy="18" rx="36" ry="14" transform="rotate(-10 520 18)" />
        <ellipse cx="760" cy="8" rx="42" ry="16" transform="rotate(15 760 8)" />
        <ellipse cx="1020" cy="16" rx="38" ry="15" transform="rotate(-14 1020 16)" />
      </g>
    </svg>
  );
}

function CanopyBottom() {
  return (
    <svg className="canopy-layer canopy-bottom back" viewBox="0 0 1200 140" preserveAspectRatio="none">
      <path fill="#0e2216" d="M0,140 L0,70 C70,30 140,90 210,55 C280,20 350,80 420,45
        C490,10 560,70 630,40 C700,10 770,65 840,35 C910,5 980,60 1050,30
        C1100,10 1150,40 1200,20 L1200,140 Z" />
    </svg>
  );
}

function CanopyBottomFront() {
  return (
    <svg className="canopy-layer canopy-bottom" viewBox="0 0 1200 100" preserveAspectRatio="none">
      <path fill="#16311f" d="M0,100 L0,45 C60,15 120,60 180,35 C240,10 300,55 360,30
        C420,5 480,50 540,25 C600,0 660,45 720,22 C780,0 840,40 900,20
        C960,2 1020,35 1080,18 C1120,8 1160,25 1200,15 L1200,100 Z" />
    </svg>
  );
}

function MascotMonkey() {
  return (
    <img
      className="mascot-monkey-img"
      src="/static/monkey%20cartoon.png"
      alt="Cartoon monkey mascot"
      width="862"
      height="862"
    />
  );
}

function Vine() {
  return (
    <>
      <svg className="vine" viewBox="0 0 400 40" preserveAspectRatio="none">
        <path d="M0 4 Q100 30 200 6 T400 8" stroke="#81a95a" strokeWidth="3" fill="none" />
        <g fill="#6fae7a">
          <ellipse cx="60" cy="14" rx="8" ry="4" transform="rotate(30 60 14)" />
          <ellipse cx="150" cy="18" rx="8" ry="4" transform="rotate(-20 150 18)" />
          <ellipse cx="260" cy="10" rx="8" ry="4" transform="rotate(25 260 10)" />
          <ellipse cx="340" cy="16" rx="8" ry="4" transform="rotate(-15 340 16)" />
        </g>
      </svg>
      <img className="vine-leaf-extra" src="/static/leaf.png" alt="" />
    </>
  );
}

function InstructionsModal({ onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="instr-title">
        <h3 id="instr-title">How reporting works</h3>
        <ol>
          <li>Start a project for the site or species group you're monitoring.</li>
          <li>Upload field data as you collect it — the report builds itself as new records come in.</li>
          <li>Review the auto-generated summary, then publish or export it whenever you're ready.</li>
        </ol>
        <button className="modal-close" onClick={onClose}>Got it</button>
      </div>
    </div>
  );
}

function ProjectRow({ project, onRemove }) {
  const isDraft = project.status === "Draft";
  return (
    <li
      className="project-row project-row-link"
      onClick={() => { window.location.href = `/projects/${project.id}`; }}
    >
      <div className="project-info">
        <div className="project-name">{project.name}</div>
        <div className="project-meta">Updated {project.updated}</div>
      </div>
      <span className={`status-pill ${isDraft ? "status-draft" : "status-progress"}`}>
        {project.status}
      </span>
      <button
        className="remove-btn"
        aria-label={`Remove ${project.name}`}
        onClick={(e) => { e.stopPropagation(); onRemove(project.id); }}
      >
        ×
      </button>
    </li>
  );
}

const WIZARD_STEPS = ["Project name", "Sites", "Time period", "Data displays", "Upload audio"];

function NewProjectModal({ onClose, onCreated }) {
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [multipleSites, setMultipleSites] = useState(false);
  const [siteNames, setSiteNames] = useState(["Site A", "Site B"]);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [audioFiles, setAudioFiles] = useState([]);
  const [audioMeta, setAudioMeta] = useState([]);
  const [displays, setDisplays] = useState(() => new Set(ALL_DISPLAY_KEYS));
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const effectiveSites = multipleSites ? siteNames.filter((s) => s.trim()) : ["Main Site"];

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const toggleDisplay = (key) => {
    setDisplays((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const handleFilesChosen = (fileList) => {
    const files = Array.from(fileList);
    setAudioFiles(files);
    setAudioMeta(files.map((_, i) => ({
      date: startDate || new Date().toISOString().slice(0, 10),
      time: "12:00",
      site: effectiveSites[i % effectiveSites.length] || "Main Site",
    })));
  };

  const updateAudioMeta = (index, field, value) => {
    setAudioMeta((prev) => prev.map((m, i) => (i === index ? { ...m, [field]: value } : m)));
  };

  const updateSiteName = (index, value) => {
    setSiteNames((prev) => prev.map((s, i) => (i === index ? value : s)));
  };

  const addSite = () => {
    setSiteNames((prev) => [...prev, `Site ${String.fromCharCode(65 + prev.length)}`]);
  };

  const removeSite = (index) => {
    setSiteNames((prev) => prev.filter((_, i) => i !== index));
  };

  const stepError = () => {
    if (step === 0 && !name.trim()) return "Give the project a name first.";
    if (step === 1 && multipleSites && effectiveSites.length < 2) return "Name at least two sites, or switch back to No.";
    if (step === 2 && (!startDate || !endDate)) return "Set a start and end date for the monitoring period.";
    return "";
  };

  const goNext = () => {
    const err = stepError();
    if (err) {
      setError(err);
      return;
    }
    setError("");
    setStep((s) => Math.min(s + 1, WIZARD_STEPS.length - 1));
  };

  const goBack = () => {
    setError("");
    setStep((s) => Math.max(s - 1, 0));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const err = stepError();
    if (err) {
      setError(err);
      return;
    }

    setSubmitting(true);
    setError("");

    const fd = new FormData();
    fd.append("name", name.trim());
    fd.append("multiple_sites", multipleSites ? "true" : "false");
    effectiveSites.forEach((s) => fd.append("site_names", s));
    fd.append("start_date", startDate);
    fd.append("end_date", endDate);
    displays.forEach((key) => fd.append("data_displays", key));
    audioFiles.forEach((file, i) => {
      fd.append("audio", file);
      const meta = audioMeta[i] || {};
      fd.append("audio_date", meta.date || startDate);
      fd.append("audio_time", meta.time || "12:00");
      fd.append("audio_site", meta.site || effectiveSites[0]);
    });

    fetch("/api/projects", { method: "POST", body: fd })
      .then((r) => {
        if (!r.ok) throw new Error("request failed");
        return r.json();
      })
      .then((project) => {
        onCreated(project);
      })
      .catch(() => setError("Couldn't create the project. Try again."))
      .finally(() => setSubmitting(false));
  };

  const isLastStep = step === WIZARD_STEPS.length - 1;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card new-project-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-project-title"
      >
        <h3 id="new-project-title">New project</h3>

        <div className="wizard-steps">
          {WIZARD_STEPS.map((label, i) => (
            <div key={label} className={`wizard-step ${i === step ? "is-active" : ""} ${i < step ? "is-done" : ""}`}>
              <span className="wizard-dot">{i < step ? "✓" : i + 1}</span>
              <span className="wizard-step-label">{label}</span>
            </div>
          ))}
        </div>

        <form onSubmit={isLastStep ? handleSubmit : (e) => { e.preventDefault(); goNext(); }}>
          {step === 0 && (
            <>
              <label className="field-label" htmlFor="np-name">Project name</label>
              <input
                id="np-name"
                autoFocus
                type="text"
                placeholder="e.g. Canopy Bird Count"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </>
          )}

          {step === 1 && (
            <>
              <div className="field-label">Multiple sites?</div>
              <div className="radio-row">
                <label>
                  <input
                    type="radio"
                    name="multiple-sites"
                    checked={!multipleSites}
                    onChange={() => setMultipleSites(false)}
                  />
                  No
                </label>
                <label>
                  <input
                    type="radio"
                    name="multiple-sites"
                    checked={multipleSites}
                    onChange={() => setMultipleSites(true)}
                  />
                  Yes
                </label>
              </div>
              {multipleSites && (
                <div className="site-names">
                  {siteNames.map((siteName, i) => (
                    <div className="site-name-row" key={i}>
                      <input
                        type="text"
                        value={siteName}
                        placeholder={`Site ${i + 1} name`}
                        onChange={(e) => updateSiteName(i, e.target.value)}
                        aria-label={`Site ${i + 1} name`}
                      />
                      {siteNames.length > 2 && (
                        <button type="button" className="remove-site-btn" aria-label={`Remove ${siteName || `site ${i + 1}`}`} onClick={() => removeSite(i)}>×</button>
                      )}
                    </div>
                  ))}
                  <button type="button" className="add-site-btn" onClick={addSite}>+ Add site</button>
                </div>
              )}
            </>
          )}

          {step === 2 && (
            <>
              <div className="field-label">Time period</div>
              <div className="date-row">
                <input
                  type="date"
                  aria-label="Start date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
                <span>to</span>
                <input
                  type="date"
                  aria-label="End date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </>
          )}

          {step === 3 && (
            <>
              <div className="field-label">Choose data displays</div>
              {DATA_DISPLAY_GROUPS.map((group) => {
                const disabled = group.requiresMultipleSites && !multipleSites;
                return (
                  <div className="display-group" key={group.title}>
                    <div className="display-group-title">
                      {group.title}
                      {disabled && <span className="display-group-hint"> (needs multiple sites)</span>}
                    </div>
                    {group.options.map((opt) => (
                      <label key={opt.key} className={`display-option ${disabled ? "is-disabled" : ""}`}>
                        <input
                          type="checkbox"
                          checked={displays.has(opt.key)}
                          disabled={disabled}
                          onChange={() => toggleDisplay(opt.key)}
                        />
                        <span>
                          {opt.label}
                          {opt.hint && !disabled && <span className="display-option-hint"> — {opt.hint}</span>}
                        </span>
                      </label>
                    ))}
                  </div>
                );
              })}
            </>
          )}

          {step === 4 && (
            <>
              <label className="field-label" htmlFor="np-audio">Upload audio</label>
              <input
                id="np-audio"
                type="file"
                accept="audio/*"
                multiple
                onChange={(e) => handleFilesChosen(e.target.files)}
              />
              <div className="muted-note">
                No audio yet? That's fine — the project is saved as a Draft and you can add audio later.
              </div>

              {audioFiles.length > 0 && (
                <div className="audio-meta-list">
                  <div className="field-label">
                    When and where was each recording made?
                  </div>
                  {audioFiles.map((file, i) => (
                    <div className="audio-meta-row" key={i}>
                      <div className="audio-meta-filename" title={file.name}>{file.name}</div>
                      <input
                        type="date"
                        value={(audioMeta[i] && audioMeta[i].date) || ""}
                        onChange={(e) => updateAudioMeta(i, "date", e.target.value)}
                        aria-label={`Date for ${file.name}`}
                      />
                      <input
                        type="time"
                        value={(audioMeta[i] && audioMeta[i].time) || ""}
                        onChange={(e) => updateAudioMeta(i, "time", e.target.value)}
                        aria-label={`Time (24-hour) for ${file.name}`}
                      />
                      {multipleSites ? (
                        <select
                          value={(audioMeta[i] && audioMeta[i].site) || ""}
                          onChange={(e) => updateAudioMeta(i, "site", e.target.value)}
                          aria-label={`Site for ${file.name}`}
                        >
                          {effectiveSites.map((s) => <option key={s} value={s}>{s}</option>)}
                        </select>
                      ) : (
                        <span className="audio-meta-site-fixed">Main Site</span>
                      )}
                    </div>
                  ))}
                  <div className="muted-note">
                    Time is 24-hour (military time) — this is what lets the report tell dawn
                    choruses from dusk calls, and line up the same moment across sites.
                  </div>
                </div>
              )}
            </>
          )}

          {error && <div className="form-error">{error}</div>}

          <div className="modal-actions">
            <button type="button" className="modal-close" onClick={step === 0 ? onClose : goBack}>
              {step === 0 ? "Cancel" : "Back"}
            </button>
            <button type="submit" className="btn-new-project" disabled={submitting}>
              {isLastStep ? (submitting ? "Creating…" : "Create project") : "Next"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function App() {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showInstructions, setShowInstructions] = useState(false);
  const [showForm, setShowForm] = useState(false);

  const loadProjects = useCallback(() => {
    fetch("/api/projects")
      .then((r) => r.json())
      .then((data) => {
        setProjects(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => { loadProjects(); }, [loadProjects]);

  const handleCreated = (project) => {
    setProjects((prev) => [...prev, project]);
    setShowForm(false);
  };

  const handleRemove = (id) => {
    setProjects((prev) => prev.filter((p) => p.id !== id));
    fetch(`/api/projects/${id}`, { method: "DELETE" }).catch(() => loadProjects());
  };

  return (
    <div className="stage">
      <div className="dapple" />
      <CanopyTop />
      <CanopyTopFront />

      <div className="content">
        <div className="hero-left">
          <h1 className="headline">Ecology Reporting</h1>
          <p className="tagline">
            Turn field observations into finished reports automatically... Start a
            project, log what you find, and let the summary write itself.
          </p>

          <div className="mascot-scene">
            <Vine />
            <div className="mascot-monkey">
              <MascotMonkey />
            </div>
            <button
              className="speech-bubble"
              onClick={() => setShowInstructions(true)}
            >
              Instructions
            </button>
          </div>
        </div>

        <div className="projects-panel">
          <h2 className="panel-title">PROJECTS</h2>
          <div className="project-list-scroll">
            {loading ? (
              <div className="empty-state">Loading projects…</div>
            ) : projects.length === 0 ? (
              <div className="empty-state">No projects yet — start one to see it here.</div>
            ) : (
              <ul className="project-list">
                {projects.map((p) => (
                  <ProjectRow key={p.id} project={p} onRemove={handleRemove} />
                ))}
              </ul>
            )}
          </div>
          <div className="new-project-row">
            <button className="btn-new-project" onClick={() => setShowForm(true)}>
              + New project
            </button>
          </div>
        </div>
      </div>

      <CanopyBottomFront />
      <CanopyBottom />

      {showInstructions && (
        <InstructionsModal onClose={() => setShowInstructions(false)} />
      )}
      {showForm && (
        <NewProjectModal onClose={() => setShowForm(false)} onCreated={handleCreated} />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
