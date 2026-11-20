# Frozen base-model setup

`append20_head.pt` is included in this repository. It is the trained 20-class
appended classifier head and requires the exact frozen 987-class BirdNET-V1
base snapshot to run.

While signed in to a GitHub account with access to this repository, download
the base snapshot from the versioned GitHub Release page:

```text
https://github.com/ejacobs5000/automated-ecology/releases/tag/birdnet-v1-base
```

Place the downloaded file at:

```text
artifacts/BirdNET_Soundscape_Model.pkl
```

The base is about 313 MB on disk. The code verifies its checksum before
inference:

```text
SHA-256  979c6133147354a87f8ee62cc0d77182259014e49aa0c5eb04d99614729a839e
```

It must match exactly: the supplied appended checkpoint was trained against
that snapshot, and a different base model invalidates the feature layout and
weights. The base is intentionally a Release asset rather than an ordinary Git
object. The copied adapter source is in `third_party/BirdNET-V1/`; its license
is included beside it.
