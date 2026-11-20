# Third-party notices

## BirdNET-V1 adapter and preprocessing

`third_party/BirdNET-V1/birdnet_demo.py` is an inference adapter based on the
BirdNET-V1 model architecture. `third_party/BirdNET-V1/audio_preprocessing.py`
contains the BirdNET-V1 spectrogram preprocessing needed to reproduce the
delivered scores. Both retain their source attribution and are distributed
with the included [BirdNET-V1 license](third_party/BirdNET-V1/LICENSE):
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0). Original authors: Stefan Kahl and the BirdNET team.

The frozen BirdNET base snapshot is distributed separately as the
`birdnet-v1-base` GitHub Release asset; see
[`artifacts/MODEL_SETUP.md`](artifacts/MODEL_SETUP.md). It is not stored as an
ordinary Git object.

## Before a public release

The project owner has approved this release path. Recipients who redistribute
the frozen base snapshot, supplied score outputs, or example-derived
information remain responsible for complying with the applicable BirdNET and
recording/data terms, and for choosing a compatible license for project
additions. Do not imply BirdNET endorsement. This file is a release checklist,
not legal advice.
