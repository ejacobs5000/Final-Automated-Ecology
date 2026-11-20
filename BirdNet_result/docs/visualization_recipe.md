# Visualisation recipe

## Recommended figure grammar

Use one panel per recording and one column per 3-second window. Each selected
class occupies a row with a fixed class colour across every panel:

- colour intensity: raw model score, darker for lower score;
- white cell outline: the score reaches that class's head-specific reporting
  threshold;
- x-axis: `0–3`, `3–6`, ..., `57–60` seconds;
- y-axis: common name, optionally scientific name in a tooltip or side table.

For a combined original-plus-appended display, show original classes first,
appended classes second, and draw a horizontal divider. State both thresholds
prominently in the subtitle. The provided `render_coverage_expansion.py`
implements this exactly.

## Useful figure choices

| Question | Input / design |
| --- | --- |
| Where are candidate events over time? | NPZ heatmap, white outlines at each head threshold |
| How does appended-label coverage add to zero-shot? | Combined heatmap; preserve original/appended divider |
| Which appended classes recur across files? | `appended_class_summary.csv` bar chart or table |
| What is a candidate event? | `candidate_detections.csv`, grouped by file, window, and head |

Avoid combining all 1,007 labels in one panel. First select labels that reach
their head-specific threshold anywhere in the chosen files; retain raw scores
for the rendered cells of those selected rows.

## Reference scripts and figures

```bash
HANDOFF=$(pwd)

# Appended 20-class pilot head only, five fixed reference recordings
python "$HANDOFF/code/render_appended_heatmap.py" \
  --input-dir "$HANDOFF/results/all_scores" \
  --output /tmp/appended_head_heatmap.png

# Original + appended candidate coverage, same five recordings
python "$HANDOFF/code/render_coverage_expansion.py" \
  --input-dir "$HANDOFF/results/all_scores" \
  --output /tmp/coverage_expansion.png
```

The expected reference renders are saved in `examples/`. Both scripts use the
same five recordings so their layouts are directly comparable.

## Wording for captions and legends

Suggested language:

> BirdNET candidate-label scores over consecutive 3-second windows. Colour
> intensity shows the raw model output; outlines mark the head-specific
> reporting threshold. The recordings are unlabeled, so labels are candidates
> rather than confirmed species detections.

For combined figures, add:

> Original zero-shot (987 classes) and appended pilot (20 classes) heads use
> different reporting thresholds; this view shows candidate-label coverage,
> not calibrated confidence or comparative accuracy.
