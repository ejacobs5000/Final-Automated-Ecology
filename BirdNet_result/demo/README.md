# Five-file BirdNET inference demo

`audio/` contains the five cleaned 60-second recordings used in the supplied
coverage-expansion figure. `raw_audio/` contains the corresponding recordings
before the supplied denoising step. Each cleaned filename has a matching score
matrix in `../results/all_scores/`.

The recordings are included only as example inputs for this handoff. They have
no species ground truth, so any output is a candidate label rather than a
confirmed detection or an accuracy measurement.

## Run the demo

Install the package requirements and download the exact frozen BirdNET snapshot
described in `../artifacts/MODEL_SETUP.md`. The snapshot is provided as a
GitHub Release asset and its SHA-256 is verified during inference.

```bash
python demo/run_example_inference.py \
  --snapshot /path/to/BirdNET_Soundscape_Model.pkl \
  --output-dir /tmp/birdnet_demo_scores
```

The command produces five `(20, 1007)` raw-score matrices under
`/tmp/birdnet_demo_scores/all_scores/`, plus candidate-event and appended-label
summary CSVs and a `run_report.json` file. It never trains or modifies either
model head.

The output uses the same split as the delivered results:

| Columns | Head | Reporting threshold |
| --- | --- | --- |
| `0:987` | frozen original BirdNET zero-shot head | `0.50` |
| `987:1007` | 20-class appended iNaturalist pilot head | `0.10` |

The thresholds are head-specific reporting points, not calibrated confidence.
Do not compare raw score magnitude across the two heads.

## Run raw-to-score inference

This variant reproduces the complete demonstration path: it denoises the five
versioned raw recordings and then runs the fixed extended model. It creates new
cleaned FLAC files and scores under the chosen work directory.

```bash
python demo/run_demo_pipeline.py \
  --snapshot /path/to/BirdNET_Soundscape_Model.pkl \
  --work-dir /tmp/birdnet_demo_raw_to_scores
```

The supplied `run_raw_to_scores.py` and `run_appended_inference.py` retain
their normal 30-file delivery check. The demo launcher passes their explicit
`--allow-subset` option because it intentionally processes exactly these five
examples.
