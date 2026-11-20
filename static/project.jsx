const { useState, useEffect, useRef, useMemo } = React;

function fmtDate(iso) {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function fmtHourBin(bin) {
  const d = new Date(bin + ":00:00");
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric" });
}

function hourLabel(h) {
  const period = h < 12 ? "am" : "pm";
  let hour12 = h % 12;
  if (hour12 === 0) hour12 = 12;
  return `${hour12}${period}`;
}

function ChartCanvas({ config, height }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const configKey = JSON.stringify(config);

  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    chartRef.current = new window.Chart(ctx, config);
    return () => chartRef.current.destroy();
  }, [configKey]);

  return (
    <div className="chart-wrap" style={{ height: height || 280 }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

function LineMultiSeries({ hours, series, title }) {
  const config = {
    type: "line",
    data: {
      labels: hours.map(fmtHourBin),
      datasets: series.map((s) => ({
        label: s.label,
        data: s.data,
        borderColor: s.color,
        backgroundColor: s.color,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { maxTicksLimit: 10, font: { size: 10 } }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  };
  return (
    <div className="chart-card">
      {title && <div className="chart-title">{title}</div>}
      <ChartCanvas config={config} />
    </div>
  );
}

function BarMultiSeries({ hours, series, title }) {
  const config = {
    type: "bar",
    data: {
      labels: hours.map(hourLabel),
      datasets: series.map((s) => ({
        label: s.label,
        data: s.data,
        backgroundColor: s.color,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { ticks: { font: { size: 10 } }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  };
  return (
    <div className="chart-card">
      {title && <div className="chart-title">{title}</div>}
      <ChartCanvas config={config} />
    </div>
  );
}

function AccumulationCurve({ data }) {
  const config = {
    type: "line",
    data: {
      labels: data.map((d) => fmtDate(d.date)),
      datasets: [{
        label: "Unique species",
        data: data.map((d) => d.cumulative),
        borderColor: "#e7a94c",
        backgroundColor: "rgba(231,169,76,0.18)",
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 10, font: { size: 10 } }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  };
  return (
    <div className="chart-card">
      <div className="chart-title">Species accumulation curve</div>
      <ChartCanvas config={config} />
    </div>
  );
}

function CalendarHeatmap({ data }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  const firstDate = new Date(data[0].date + "T00:00:00");
  const leadingBlanks = firstDate.getDay();
  const cells = [...Array(leadingBlanks).fill(null), ...data];

  const colorFor = (count) => {
    if (count === 0) return "rgba(14,34,22,0.06)";
    const ratio = count / max;
    const alpha = 0.25 + ratio * 0.65;
    return `rgba(63,136,98,${alpha.toFixed(2)})`;
  };

  return (
    <div className="chart-card">
      <div className="chart-title">Calendar heatmap</div>
      <div className="calendar-weekdays">
        {["S", "M", "T", "W", "T", "F", "S"].map((label, i) => (
          <span key={i}>{label}</span>
        ))}
      </div>
      <div className="calendar-grid">
        {cells.map((cell, i) => (
          <div
            key={i}
            className="calendar-cell"
            style={{ background: cell ? colorFor(cell.count) : "transparent" }}
            title={cell ? `${fmtDate(cell.date)}: ${cell.count} detection${cell.count === 1 ? "" : "s"}` : ""}
          >
            {cell && <span className="calendar-cell-day">{new Date(cell.date + "T00:00:00").getDate()}</span>}
          </div>
        ))}
      </div>
      <div className="calendar-legend">
        <span>Less</span>
        {[0, 0.25, 0.5, 0.75, 1].map((r) => (
          <div key={r} className="calendar-swatch" style={{ background: colorFor(r * max || (r === 0 ? 0 : 1)) }} />
        ))}
        <span>More</span>
      </div>
    </div>
  );
}

function TaxonomicBreakdown({ groups }) {
  return (
    <div className="chart-card">
      <div className="chart-title">Taxonomic breakdown (by family)</div>
      <div className="taxonomy-groups">
        {groups.map((g) => (
          <div key={g.family_name} className="taxonomy-group">
            <div className="taxonomy-class">{g.family_name}</div>
            <ul>
              {g.species.map((s) => (
                <li key={s.key}>{s.label} <span className="muted">({s.count})</span></li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

function Leaderboard({ items }) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <div className="chart-card">
      <div className="chart-title">Species leaderboard</div>
      <ol className="leaderboard-list">
        {items.map((item) => (
          <li key={item.key}>
            <span className="leaderboard-rank">#{item.rank}</span>
            <span className="leaderboard-name">{item.label}</span>
            <div className="leaderboard-bar-track">
              <div className="leaderboard-bar" style={{ width: `${(item.count / max) * 100}%` }} />
            </div>
            <span className="leaderboard-count">{item.count}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function NewSpeciesFlagged({ items }) {
  return (
    <div className="chart-card">
      <div className="chart-title">New species flagged</div>
      {items.length === 0 ? (
        <div className="muted">No newly-appearing species in the back quarter of this project's timeline.</div>
      ) : (
        <ul className="plain-list">
          {items.map((item) => (
            <li key={item.key}>
              <strong>{item.label}</strong> — first heard {fmtDate(item.first_seen)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function NotableRare({ items }) {
  return (
    <div className="chart-card">
      <div className="chart-title">Notable / rare species</div>
      {items.length === 0 ? (
        <div className="muted">No species of elevated conservation concern detected.</div>
      ) : (
        <div className="notable-list">
          {items.map((item) => (
            <div key={item.key} className="notable-callout">
              <strong>{item.label}</strong>
              <span className="status-chip">{item.status}</span>
              <span className="muted">{item.count} detection{item.count === 1 ? "" : "s"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SpeciesLookup({ items }) {
  const [query, setQuery] = useState("");
  const filtered = items.filter((s) => {
    const q = query.toLowerCase();
    return s.common_name.toLowerCase().includes(q) || s.scientific_name.toLowerCase().includes(q);
  });
  return (
    <div className="chart-card">
      <div className="chart-title">Species lookup</div>
      <input
        className="lookup-search"
        type="text"
        placeholder="Search species…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="lookup-table-wrap">
        <table className="lookup-table">
          <thead>
            <tr>
              <th>Common name</th>
              <th>Scientific name</th>
              <th>Class</th>
              <th>
                Status
                <span className="info-tip" tabIndex="0" title="Status comes from the IUCN Red List via Wikidata, looked up by scientific name. 'Not assessed' means that lookup found no match (e.g. offline, or the species isn't in Wikidata) — it is not a claim that the species is unthreatened.">ⓘ</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((s) => (
              <tr key={s.key}>
                <td>{s.common_name}</td>
                <td className="italic">{s.scientific_name}</td>
                <td>{s.class_name}</td>
                <td>{s.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function AddAudioPanel({ project, onAdded }) {
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState([]);
  const [meta, setMeta] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const today = new Date().toISOString().slice(0, 10);

  const chooseFiles = (fileList) => {
    const chosen = Array.from(fileList);
    setFiles(chosen);
    setMeta(chosen.map(() => ({
      date: today, time: "12:00", site: project.sites[0], addingSite: false, newSiteName: "",
    })));
  };

  const updateMeta = (i, field, value) => {
    setMeta((prev) => prev.map((m, idx) => (idx === i ? { ...m, [field]: value } : m)));
  };

  const submit = (e) => {
    e.preventDefault();
    if (files.length === 0) {
      setError("Choose at least one audio file first.");
      return;
    }
    const resolvedSites = meta.map((m) => (m.addingSite ? m.newSiteName.trim() : m.site));
    if (resolvedSites.some((s) => !s)) {
      setError("Enter a name for each new site, or switch back to an existing one.");
      return;
    }
    setSubmitting(true);
    setError("");
    const fd = new FormData();
    files.forEach((file, i) => {
      fd.append("audio", file);
      fd.append("audio_date", meta[i].date);
      fd.append("audio_time", meta[i].time);
      fd.append("audio_site", resolvedSites[i]);
    });
    fetch(`/api/projects/${project.id}/audio`, { method: "POST", body: fd })
      .then((r) => {
        if (!r.ok) throw new Error("request failed");
        return r.json();
      })
      .then(() => {
        setFiles([]);
        setMeta([]);
        setOpen(false);
        onAdded();
      })
      .catch(() => setError("Couldn't add that audio. Try again."))
      .finally(() => setSubmitting(false));
  };

  if (!open) {
    return (
      <div className="add-audio-row no-print">
        <button type="button" className="btn-new-project" onClick={() => setOpen(true)}>
          + Add audio to this project
        </button>
      </div>
    );
  }

  return (
    <div className="add-audio-row no-print">
      <form className="add-audio-form" onSubmit={submit}>
        <label className="field-label" htmlFor="add-audio-input">Add audio</label>
        <input
          id="add-audio-input"
          type="file"
          accept="audio/*"
          multiple
          onChange={(e) => chooseFiles(e.target.files)}
        />
        {files.length > 0 && (
          <div className="audio-meta-list">
            {files.map((file, i) => (
              <div className="audio-meta-row" key={i}>
                <div className="audio-meta-filename" title={file.name}>{file.name}</div>
                <input
                  type="date"
                  value={meta[i].date}
                  onChange={(e) => updateMeta(i, "date", e.target.value)}
                  aria-label={`Date for ${file.name}`}
                />
                <input
                  type="time"
                  value={meta[i].time}
                  onChange={(e) => updateMeta(i, "time", e.target.value)}
                  aria-label={`Time (24-hour) for ${file.name}`}
                />
                {meta[i].addingSite ? (
                  <div className="audio-meta-site-new">
                    <input
                      type="text"
                      placeholder="New site name"
                      value={meta[i].newSiteName}
                      onChange={(e) => updateMeta(i, "newSiteName", e.target.value)}
                      aria-label={`New site name for ${file.name}`}
                    />
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => updateMeta(i, "addingSite", false)}
                    >
                      Use existing site
                    </button>
                  </div>
                ) : (
                  <div className="audio-meta-site-existing">
                    <select
                      value={meta[i].site}
                      onChange={(e) => updateMeta(i, "site", e.target.value)}
                      aria-label={`Site for ${file.name}`}
                    >
                      {project.sites.map((s) => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => updateMeta(i, "addingSite", true)}
                    >
                      + New site
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {error && <div className="form-error">{error}</div>}
        <div className="modal-actions">
          <button type="button" className="modal-close" onClick={() => setOpen(false)}>Cancel</button>
          <button type="submit" className="btn-new-project" disabled={submitting}>
            {submitting ? "Adding…" : "Add to project"}
          </button>
        </div>
      </form>
    </div>
  );
}

function AudioDetectionRow({ projectId, detection, speciesList, onChanged }) {
  const [busy, setBusy] = useState(false);

  const override = (speciesKey) => {
    if (speciesKey === detection.species_key) return;
    setBusy(true);
    fetch(`/api/projects/${projectId}/detections/${detection.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ species_key: speciesKey }),
    })
      .then(onChanged)
      .finally(() => setBusy(false));
  };

  const reject = () => {
    if (!window.confirm(`Remove this detection (${detection.label})? This can't be undone.`)) return;
    setBusy(true);
    fetch(`/api/projects/${projectId}/detections/${detection.id}`, { method: "DELETE" })
      .then(onChanged)
      .finally(() => setBusy(false));
  };

  return (
    <div className={`audio-detection-row ${busy ? "is-busy" : ""}`}>
      <select
        value={detection.species_key}
        disabled={busy}
        onChange={(e) => override(e.target.value)}
        aria-label={`Species for detection at ${fmtTime(detection.timestamp)}`}
      >
        {!speciesList.some((s) => s.key === detection.species_key) && (
          <option value={detection.species_key}>{detection.label}</option>
        )}
        {speciesList.map((s) => (
          <option key={s.key} value={s.key}>{s.common_name}</option>
        ))}
      </select>
      {detection.user_overridden && <span className="overridden-chip">edited</span>}
      <span className="audio-detection-time">{fmtTime(detection.timestamp)}</span>
      <span className="audio-detection-confidence">{Math.round(detection.confidence * 100)}%</span>
      <button className="reject-btn" disabled={busy} onClick={reject} aria-label={`Reject ${detection.label} detection`}>×</button>
    </div>
  );
}

function AudioFileCard({ projectId, file, speciesList, onChanged }) {
  const [busy, setBusy] = useState(false);

  const removeRecording = () => {
    if (!window.confirm(`Remove "${file.filename}" from this project? This can't be undone.`)) return;
    setBusy(true);
    fetch(`/api/projects/${projectId}/audio/${encodeURIComponent(file.filename)}`, { method: "DELETE" })
      .then(onChanged)
      .finally(() => setBusy(false));
  };

  return (
    <div className={`audio-file-card ${busy ? "is-busy" : ""}`}>
      <div className="audio-file-header">
        <div className="audio-file-header-info">
          <div className="audio-file-name" title={file.filename}>{file.filename}</div>
          <div className="audio-file-meta">{file.site} · {fmtDate(file.date)} {file.time}</div>
        </div>
        <button
          type="button"
          className="remove-audio-btn"
          disabled={busy}
          onClick={removeRecording}
          aria-label={`Remove ${file.filename} from this project`}
        >
          Remove
        </button>
      </div>
      {file.detections.length === 0 ? (
        <div className="muted">No detections in this recording.</div>
      ) : (
        <div className="audio-detection-list">
          {file.detections.map((det) => (
            <AudioDetectionRow
              key={det.id}
              projectId={projectId}
              detection={det}
              speciesList={speciesList}
              onChanged={onChanged}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function PerAudioSection({ projectId, perAudio, analysisSource, speciesList, onChanged }) {
  return (
    <section className="dashboard-section no-print">
      <h2 className="section-title">Per recording</h2>
      {analysisSource !== "real" && (
        <div className="audio-section-note">
          No real BirdNET analysis has run for this project yet — the BirdNET model isn't
          installed on this server, or the uploaded audio couldn't be analyzed. See the
          note above for why.
        </div>
      )}
      <div className="audio-file-list">
        {perAudio.map((file) => (
          <AudioFileCard
            key={file.filename}
            projectId={projectId}
            file={file}
            speciesList={speciesList}
            onChanged={onChanged}
          />
        ))}
      </div>
    </section>
  );
}

function Section({ title, children }) {
  return (
    <section className="dashboard-section">
      <h2 className="section-title">{title}</h2>
      <div className="section-grid">{children}</div>
    </section>
  );
}

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

function EditReportModal({ project, onClose, onSaved }) {
  const [displays, setDisplays] = useState(() => new Set(project.data_displays));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

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

  const save = () => {
    setSaving(true);
    setError("");
    fetch(`/api/projects/${project.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_displays: Array.from(displays) }),
    })
      .then((r) => {
        if (!r.ok) throw new Error("request failed");
        onSaved();
        onClose();
      })
      .catch(() => setError("Couldn't save your changes. Try again."))
      .finally(() => setSaving(false));
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card new-project-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="edit-report-title"
      >
        <h3 id="edit-report-title">Edit report</h3>
        <div className="field-label">Choose data displays</div>
        {DATA_DISPLAY_GROUPS.map((group) => {
          const disabled = group.requiresMultipleSites && !project.multiple_sites;
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

        {error && <div className="form-error">{error}</div>}

        <div className="modal-actions">
          <button type="button" className="modal-close" onClick={onClose}>Cancel</button>
          <button type="button" className="btn-new-project" disabled={saving} onClick={save}>
            {saving ? "Saving…" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}

function App() {
  const projectId = window.location.pathname.split("/").filter(Boolean).pop();
  const [project, setProject] = useState(null);
  const [error, setError] = useState(false);
  const [speciesList, setSpeciesList] = useState([]);
  const [showEditReport, setShowEditReport] = useState(false);

  const loadProject = () => {
    fetch(`/api/projects/${projectId}`)
      .then((r) => {
        if (!r.ok) throw new Error("not found");
        return r.json();
      })
      .then((p) => {
        setProject(p);
        document.title = p.name; // so the browser's print header shows the project, not the generic page title
      })
      .catch(() => setError(true));
  };

  useEffect(loadProject, [projectId]);

  useEffect(() => {
    fetch("/api/species")
      .then((r) => r.json())
      .then(setSpeciesList)
      .catch(() => {});
  }, []);

  if (error) {
    return (
      <div className="project-page">
        <div className="project-empty">Project not found. <a href="/">Back to projects</a></div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="project-page">
        <div className="project-empty">Loading project…</div>
      </div>
    );
  }

  const displays = new Set(project.data_displays);
  const d = project.dashboard;
  const hasData = d.detection_count > 0;

  const showPerSpecies = ["ts_per_species", "diel_per_species", "calendar_heatmap"].some((k) => displays.has(k));
  const showSummative = ["taxonomic_breakdown", "species_leaderboard", "species_accumulation", "new_species_flagged", "notable_rare", "species_lookup"].some((k) => displays.has(k));
  const showComparisons = project.multiple_sites && d.comparisons && ["comparison_timeseries", "comparison_diel"].some((k) => displays.has(k));

  return (
    <div className="project-page">
      <header className="project-header">
        <div className="header-top-row">
          <a className="back-link" href="/">← Back to projects</a>
          <div className="header-actions no-print">
            <button type="button" className="print-btn" onClick={() => setShowEditReport(true)}>
              ✎ Edit report
            </button>
            <button type="button" className="print-btn" onClick={() => window.print()}>
              🖨 Print / Save as PDF
            </button>
          </div>
        </div>
        <h1>{project.name}</h1>
        <div className="project-meta-row">
          <span className={`status-pill ${project.status === "Draft" ? "status-draft" : "status-progress"}`}>
            {project.status}
          </span>
          {project.analysis_source === "real" && (
            <span className="source-pill source-real" title="Detections came from the real BirdNET model">
              Real BirdNET analysis
            </span>
          )}
          {project.analysis_source !== "real" && project.analysis_source !== "sample" && project.status !== "Draft" && (
            <span className="source-pill source-synthetic" title="This project has uploaded audio but no real BirdNET analysis has run for it — see the note below for why">
              No analysis yet
            </span>
          )}
          <span>{project.sites.join(", ")}</span>
          <span>{fmtDate(project.start_date)} – {fmtDate(project.end_date)}</span>
          <span>{d.detection_count} detections · {d.species_richness} species</span>
        </div>
        {project.analysis_errors && project.analysis_errors.length > 0 && (
          <div className="analysis-error-note">
            {project.analysis_source === "real" ? "Some uploads couldn't be analyzed: " : "Couldn't run real analysis: "}
            {project.analysis_errors.join(" · ")}
          </div>
        )}
      </header>

      <AddAudioPanel project={project} onAdded={loadProject} />

      {d.per_audio && d.per_audio.length > 0 && (
        <PerAudioSection
          projectId={project.id}
          perAudio={d.per_audio}
          analysisSource={project.analysis_source}
          speciesList={speciesList}
          onChanged={loadProject}
        />
      )}

      {!hasData ? (
        <div className="project-empty">
          No detections yet — upload audio for this project to populate the report.
        </div>
      ) : (
        <>
          {showPerSpecies && (
            <Section title="Per species">
              {displays.has("ts_per_species") && (
                <LineMultiSeries hours={d.ts_per_species.hours} series={d.ts_per_species.series} title="Time vs. species spotting" />
              )}
              {displays.has("diel_per_species") && (
                <BarMultiSeries hours={d.diel_per_species.hours} series={d.diel_per_species.series} title="Diel activity by hour" />
              )}
              {displays.has("calendar_heatmap") && <CalendarHeatmap data={d.calendar_heatmap} />}
            </Section>
          )}

          {showSummative && (
            <Section title="Summative species info">
              {displays.has("taxonomic_breakdown") && <TaxonomicBreakdown groups={d.taxonomic_breakdown} />}
              {displays.has("species_leaderboard") && <Leaderboard items={d.leaderboard} />}
              {displays.has("species_accumulation") && <AccumulationCurve data={d.species_accumulation} />}
              {displays.has("new_species_flagged") && <NewSpeciesFlagged items={d.new_species_flagged} />}
              {displays.has("notable_rare") && <NotableRare items={d.notable_rare} />}
              {displays.has("species_lookup") && <SpeciesLookup items={d.species_lookup} />}
            </Section>
          )}

          {showComparisons && (
            <Section title="Comparisons">
              {displays.has("comparison_timeseries") && (
                <LineMultiSeries hours={d.comparisons.timeseries.hours} series={d.comparisons.timeseries.series} title="Time vs. species spotting, by site" />
              )}
              {displays.has("comparison_diel") && (
                <BarMultiSeries hours={d.comparisons.diel.hours} series={d.comparisons.diel.series} title="Diel activity by hour, by site" />
              )}
            </Section>
          )}
        </>
      )}

      {showEditReport && (
        <EditReportModal
          project={project}
          onClose={() => setShowEditReport(false)}
          onSaved={loadProject}
        />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
