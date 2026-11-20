# Reproduction and provenance

## Required components

The included `artifacts/append20_head.pt` is the trainable component of the
final model: a 20-class `1 x 1` convolution plus metadata. It is not a
standalone 1,007-output network. Pair it with the exact frozen 987-class
BirdNET-V1 snapshot described in
[`../artifacts/MODEL_SETUP.md`](../artifacts/MODEL_SETUP.md).

The inference loader checks this SHA-256 before loading:

```text
979c6133147354a87f8ee62cc0d77182259014e49aa0c5eb04d99614729a839e
```

The repository includes the two small BirdNET-dependent source files required
to recreate the preprocessing and frozen-backbone adapter in
`third_party/BirdNET-V1/`. Their license is included in the same directory.
Install the Python dependencies listed in `requirements.txt`.

## Verify the delivered matrices

```bash
python code/validate_visualization_input.py \
  --input-dir results/all_scores
```

Expected result: 30 files, 600 windows, 1,007 labels, 79 original-head events,
and 113 appended-head events.

## Re-run deployment: raw audio to scores

Training is deliberately absent from the deployment path. Supply a directory
with the 30 authorized raw 60-second FLAC recordings and fresh output
directories:

```bash
HANDOFF=$(pwd)
python code/run_raw_to_scores.py \
  --raw-input-dir /path/to/raw_flac \
  --clean-output-dir /tmp/birdnet_extended_recompute/cleaned \
  --score-output-dir /tmp/birdnet_extended_recompute/scores \
  --snapshot "$HANDOFF/artifacts/BirdNET_Soundscape_Model.pkl" \
  --device cpu --batch-size 8 --threads 4
```

The wrapper first invokes `code/denoise_pipeline.py`, then invokes
`code/run_appended_inference.py` using the fixed checkpoint. It verifies the
model layout and base-model checksum. If an optional compatible zero-shot
reference directory is passed to `run_appended_inference.py`, the unchanged
987 original columns are also compared against it.

## Training provenance (not a deployment step)

`code/finetune_append20.py` and the reports in `artifacts/` document how the
append head was made. The pilot used 1,000 weakly labelled recordings split as
700 train, 140 validation, and 160 held-out test examples. Those data and the
training commands are retained for provenance only; they are not needed for
visualization or routine raw-audio deployment.
