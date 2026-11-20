# Extended BirdNET visualization handoff

This repository is the deployment and visualization handoff for the extended
BirdNET-V1 model. Its fixed operational path is:

```text
raw FLAC -> denoising -> cleaned FLAC -> 3-second BirdNET windows -> 1,007 raw scores -> visualization
```

It contains **inference only**. The appended model head has already been
trained; routine use must not train, tune, or alter any model weights.

## Contents

1. [BirdNET-V1 base-model release](https://github.com/ejacobs5000/automated-ecology/releases/tag/birdnet-v1-base)
   — exact frozen 987-class BirdNET-V1 snapshot required for inference;
   download its asset while signed in to a GitHub account with repository access.
2. `artifacts/append20_head.pt` — trained 20-class appended head and its
   metadata.
3. `results/all_scores/` — delivered raw scores for 30 unlabeled 60-second
   recordings: 30 files x 20 windows x 1,007 labels.
4. `results/class_index_map.csv` — label, output-column index, model head, and
   reporting threshold for all 1,007 scores.
5. `code/` — raw-audio denoising, fixed-model inference, validation, and
   reusable visualization scripts.
6. `third_party/BirdNET-V1/` — small adapter and preprocessing source needed
   for inference, preserved with its upstream license.
7. `docs/` — method, data contract, visualization recipe, and provenance.
8. `demo/` — five matched raw/cleaned audio examples, plus direct-inference and
   raw-to-score demo launchers.

The frozen base is published as a separate GitHub Release asset so the
repository remains lightweight. Collaborators with repository access download
it once from the signed-in GitHub Release page, place it in `artifacts/`, and
pair it with the included `append20_head.pt` to run the complete fixed
1,007-output model: 987 original BirdNET outputs plus 20 appended pilot
outputs. See [`artifacts/MODEL_SETUP.md`](artifacts/MODEL_SETUP.md). Raw audio
is not committed.

## Install and configure

```bash
git clone <your-repository-url>
cd model_visualization_handoff

python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Open the [BirdNET-V1 base-model Release](https://github.com/ejacobs5000/automated-ecology/releases/tag/birdnet-v1-base)
while signed in to a GitHub account with access to this repository. Download
`BirdNET_Soundscape_Model.pkl` from **Assets**, then place it at
`artifacts/BirdNET_Soundscape_Model.pkl` and verify it before inference:

```bash
sha256sum artifacts/BirdNET_Soundscape_Model.pkl
```

The expected SHA-256 is
`979c6133147354a87f8ee62cc0d77182259014e49aa0c5eb04d99614729a839e`.
The inference loader also checks this value and will reject a different model.
See [`artifacts/MODEL_SETUP.md`](artifacts/MODEL_SETUP.md) for the release
link and licensing notice.

## Start with the delivered scores

Visualization work normally needs no model run. Load the matrices from
`results/all_scores/` using `numpy.load(path, allow_pickle=False)` and follow
the schema in [`docs/data_contract.md`](docs/data_contract.md).

| Need | Use |
| --- | --- |
| Every raw score, including scores below a threshold | `results/all_scores/*_all_scores.npz` |
| Label/head/threshold for every score column | `results/class_index_map.csv` |
| Thresholded candidate events | `results/candidate_detections.csv` |
| Appended-class overview | `results/appended_class_summary.csv` |
| Reference graphics and renderer scripts | `examples/` and `code/render_*.py` |

`candidate_detections.csv` is a thresholded summary, not the source for an
all-score heatmap. The NPZ matrices are the source of truth for graphics.

Each supplied matrix has shape `(20, 1007)`: 20 consecutive, non-overlapping
3-second windows by 1,007 candidate labels. It stores `start_seconds`,
`scores`, `classes`, `source_file`, and the two reporting thresholds.

## Interpretation rules

- The supplied recordings have no species ground truth. Describe outputs as
  *candidate detections* or *model outputs*, not confirmed detections,
  probabilities, or accuracy.
- Scores are raw, uncalibrated sigmoid outputs. They are useful for
  within-class intensity displays, but are not calibrated confidence values.
- Columns `0–986` are the frozen original BirdNET zero-shot head; use its
  reporting threshold of `0.50`.
- Columns `987–1006` are the fine-tuned appended pilot head; use its
  validation-selected reporting threshold of `0.10`.
- Do not compare raw score magnitudes across heads. A plot may use
  `raw_score / head_threshold`, clipped to `[0, 1]`, but must label it
  *relative-to-threshold strength*. That is a visualization transform only: it
  does not change stored scores, thresholds, or model outputs.

```python
with np.load(score_file, allow_pickle=False) as data:
    scores = data["scores"]
    classes = data["classes"]

original_scores = scores[:, :987]
appended_scores = scores[:, 987:]
```

## Raw audio to scores

The included deployment wrapper implements the exact workflow used for this
release: denoise source FLAC, then run fixed-model inference. It expects the
same campaign structure as the delivered data — exactly 30 raw 60-second FLAC
files — and writes only to fresh output directories to avoid mixing runs.

```bash
HANDOFF=$(pwd)
RAW=/path/to/30_raw_flac_files
WORK=/tmp/birdnet_raw_to_scores

python "$HANDOFF/code/run_raw_to_scores.py" \
  --raw-input-dir "$RAW" \
  --clean-output-dir "$WORK/cleaned" \
  --score-output-dir "$WORK/scores" \
  --denoise-report "$WORK/denoise_report.csv" \
  --snapshot "$HANDOFF/artifacts/BirdNET_Soundscape_Model.pkl" \
  --device cpu --batch-size 8 --threads 4
```

This generates `all_scores/*.npz`, `candidate_detections.csv`,
`appended_class_summary.csv`, `original_score_regression.csv`, and
`run_report.json` beneath `$WORK/scores`. Use `--dry-run` to inspect the two
commands without processing audio.

For each file, the code decodes FLAC, averages any channels to mono, resamples
to the sample rate stored in the frozen snapshot, takes complete
non-overlapping 3-second windows, transforms them to the original BirdNET
`64 x 384` spectrogram, and produces all 1,007 raw sigmoid scores. See
[`docs/method.md`](docs/method.md) for the architecture and
[`docs/reproducibility.md`](docs/reproducibility.md) for provenance.

## Five-file audio demo

`demo/` packages the five recordings shown in the supplied coverage-expansion
figure, both before and after denoising. After downloading the base snapshot as
described above, reproduce the full raw-audio path on just those examples:

```bash
python demo/run_demo_pipeline.py \
  --snapshot /path/to/BirdNET_Soundscape_Model.pkl \
  --work-dir /tmp/birdnet_demo_raw_to_scores
```

See [`demo/README.md`](demo/README.md) for direct inference on the cleaned
examples and output details.

## Verify the delivered package

```bash
python code/validate_visualization_input.py \
  --input-dir results/all_scores

python code/render_coverage_expansion.py \
  --input-dir results/all_scores \
  --output /tmp/coverage_expansion.png
```

The validation command checks the full delivery: 30 matrices, 600 windows,
1,007 labels, metadata, class order, thresholds, and the expected event totals
(79 original-head and 113 appended-head candidate events).

## Release notes

The base snapshot is distributed as the versioned `birdnet-v1-base` GitHub
Release asset, not as an ordinary Git object. Review
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the accompanying license
terms before redistributing it or the supplied score outputs.
