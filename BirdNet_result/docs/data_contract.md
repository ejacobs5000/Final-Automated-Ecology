# Visualisation data contract

## Primary input: one NPZ file per recording

`results/all_scores/<recording-stem>_all_scores.npz` is the canonical input
for score visualisations. Files are NumPy compressed archives and must be
opened with `allow_pickle=False`.

```python
with np.load(path, allow_pickle=False) as data:
    starts = data["start_seconds"]   # float64, shape (20,)
    scores = data["scores"]          # float32, shape (20, 1007)
    labels = data["classes"]         # Unicode strings, shape (1007,)
```

All files have the following required fields:

| Field | Type / shape | Meaning |
| --- | --- | --- |
| `start_seconds` | `float64`, `(20,)` | Window start times: 0, 3, ..., 57 |
| `scores` | `float32`, `(20, 1007)` | Raw sigmoid score, `[window, class]` |
| `classes` | Unicode, `(1007,)` | Aligned label for every score column |
| `source_file` | scalar Unicode | Original FLAC basename |
| `original_report_threshold` | scalar `float32` | `0.50` |
| `appended_report_threshold` | scalar `float32` | `0.10` |
| `checkpoint` | scalar Unicode | Checkpoint used for appended head |

Rows are windows; columns are classes. `start_seconds[i]` labels the interval
`[start_seconds[i], start_seconds[i] + 3)`.

## Class layout

| Column range | Head | Reporting threshold | Meaning |
| --- | --- | --- | --- |
| `0:987` | `original_zero_shot` | `>= 0.50` | Frozen BirdNET-V1 labels |
| `987:1007` | `appended_pilot` | `>= 0.10` | Newly trained pilot labels |

Every label follows `Scientific name_Common name`, split at the first
underscore. Preserve the supplied class order; do not assume alphabetical order
or build a new index from the labels.

`results/class_index_map.csv` is the ready-to-use version of this mapping. It
has one row for each of the 1,007 columns and provides `class_index`, `head`,
`scientific_name`, `common_name`, `label`, and `reporting_threshold`.

## Tidy summaries

### `candidate_detections.csv`

One row exists only when a score reaches its own head's reporting threshold.
It has these columns:

```text
filename, window_index, start_s, end_s, head, class_index,
scientific_name, common_name, score, reporting_threshold
```

Use it for event lists, tables, and markers. Its absence for a class/window
means “below the reporting threshold,” not necessarily score zero.

### `appended_class_summary.csv`

This is a 20-row summary for appended labels, sorted by event count then peak
score. `peak_global_window_index` is zero-based in the concatenation of
alphabetically sorted recording files, 20 windows per file. It is not a time
index within a single recording.

### `original_score_regression.csv`

This is an integrity check for five files shared with the earlier zero-shot
run. It verifies that the first 987 scores and label order remain unchanged;
it is not a performance evaluation.

### `run_report.json`

Contains the final counts, provenance, thresholds, and caveat. Use it for
figure captions and dashboard metadata rather than retyping constants.

## Safe score-to-visual mapping

For a heatmap, select classes using a head-specific condition, but render each
selected cell using its raw score. For example:

```python
original_ids = np.flatnonzero(scores[:, :987].max(axis=0) >= 0.50)
appended_ids = 987 + np.flatnonzero(scores[:, 987:].max(axis=0) >= 0.10)
```

A consistent hue per class and score-driven opacity/intensity works well.
White outlines can mark cells that pass the applicable threshold. Never use a
single threshold across both heads and never interpret the score scale as a
probability or cross-head calibrated confidence.
