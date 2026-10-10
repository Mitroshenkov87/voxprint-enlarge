# Next iteration

These items are the next pass after `v0.1.0-rc.1`. They are not in this build, and this document does not schedule them.

## CLI parity

Everything the window can do should be available with no window open.

* One headless command covers add, model choice, scale, format, quality, lossless JPEG XL, output folder, batch (a folder and its subfolders), cancel, and resume.
* `--json` prints one JSON object per event on stdout, the same way `--json-progress` already does for a single Lucent job.
* Exit codes are stable and documented: success, bad arguments, GPU missing or too old, model missing, a failed item, and cancel.
* The reference is `docs/CLI.md`, including the codes and one example of each command.

The window and the current `--upscale` and `--json-progress` commands stay as they are until that document exists.

## VRAM-aware model tiers

The shared `voxprint_engine` registry should know what a card can run.

* Two tiers: 16 GB, and 24 GB or more, with 2 to 3 GB of headroom left unused.
* The registry records which tier a model and a scale need. The app measures the card, picks the tier that fits, and hides options the card cannot run.
* An advanced override shows the hidden options and lets the operator choose one anyway.

## Multi-source model downloads

A pinned model should list more than one place to download the same file.

* Each source is a URL. Every source for one file has the same sha256.
* The download tries the sources in order and moves to the next when a source fails. A hash mismatch is a failure, not a success.
* The list includes our Hugging Face mirror and a Kaggle mirror, plus the upstream URL already in the registry.
* The Model step shows which source supplied the file. **Check hash** still compares the bytes on disk with the one pinned digest.
