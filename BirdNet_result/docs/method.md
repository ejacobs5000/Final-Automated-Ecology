# Method: frozen BirdNET-V1 plus a 20-class appended head

## Purpose

The project expands the candidate-label coverage of a legacy BirdNET-V1 model
without changing its existing 987 outputs. It is a pilot fine-tuning
experiment, followed by descriptive inference on unlabeled audio. It is not a
species-identification accuracy study for the Denoising recordings.

## Input and windowing

Each input FLAC is decoded to floating-point audio, stereo is averaged to
mono, and audio is resampled to the rate encoded in the BirdNET snapshot. The
model consumes full, non-overlapping 3-second windows. The delivered recordings
are 60 seconds long, so every score matrix has 20 windows beginning at 0, 3,
..., 57 seconds. Incomplete final windows would be omitted.

Each window passes through the original BirdNET preprocessing and becomes a
`64 x 384` spectrogram. No geographic or seasonal filter is applied.

## Model architecture

```text
3-second audio → BirdNET spectrogram → frozen BirdNET-V1 backbone
                                     ↓
                         feature layer 153 (1,024 channels)
                              ↙                       ↘
unchanged original 987-class 1x1 head       trained new 20-class 1x1 head
                              ↘                       ↙
                      concatenate → 1,007 sigmoid scores
```

The original BirdNET snapshot, all layers before the last classifier, and its
987-class output weights are frozen. The original scores are reproduced by the
immutable weights. A new 20-filter `1 x 1` convolution is trained on the
1,024-channel feature tensor at layer 153. Both heads use the same
log-mean-exp temporal pooling used by BirdNET, followed by sigmoid scores.

The 20-class head applies dropout (`p=0.5`) while training; dropout is off at
inference.

## Pilot training

The pilot uses 20 iNaturalist species with 50 recordings each (1,000 WAVs).
Labels are weak, recording-level labels rather than timestamped calls. A
deterministic class-stratified split puts 35 recordings per species in train,
7 in validation, and 8 in test (700 / 140 / 160 recordings).

Only the appended head is optimised. Binary cross-entropy is calculated over
the 20 appended labels; the pre-existing 987 classes are treated as
unobserved, **not negative**. This prevents the weak pilot labels from teaching
the model that original BirdNET species are absent.

Training configuration:

- Adam, learning rate `0.001`, no weight decay
- maximum 80 epochs, early-stopping patience 12
- `ReduceLROnPlateau` on validation BCE (factor 0.5, patience 3)
- deterministic seed `365`
- one fixed 3-second crop per pilot recording

The delivered checkpoint is `artifacts/append20_head.pt`. It was selected at
epoch 80. Validation selected the appended threshold `0.10` by macro-F1; test
top-1 accuracy was 55.0% and test macro-F1 was 0.3939. These values measure
only the internal weak-label pilot holdout and must not be transferred to the
Denoising data.

## Descriptive inference run

The delivered results cover 30 unlabeled `cleaned_v4` FLACs:

- 600 total windows (`30 x 20`)
- 79 candidate events from the original 987-class head at threshold `0.50`
- 113 candidate events from the appended head at threshold `0.10`
- 16 of 20 appended labels have at least one candidate event

For five files previously analysed with zero-shot BirdNET, the preserved
original 987 score columns were checked against prior outputs. The largest
absolute difference was `1.1641532182693481e-10`, under the required `1e-6`
tolerance.

## Scope and limitations

The Denoising recordings are unlabeled, so figures should describe model score
patterns or candidate-label coverage only. They cannot show true positives,
false positives, precision, recall, or accuracy for these recordings.

Thresholds are operating points for reporting, not calibrated probabilities:

- Original zero-shot labels (`class_index < 987`): `score >= 0.50`
- Appended pilot labels (`class_index >= 987`): `score >= 0.10`

The different thresholds reflect different heads and training histories. A
score of 0.10 in the appended head must not be portrayed as equivalent to a
score of 0.10 or 0.50 in the original head.
