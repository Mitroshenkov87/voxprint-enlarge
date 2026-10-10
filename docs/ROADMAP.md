# Next iteration

These items are still ahead of the headless command in `docs/CLI.md`. This document does not schedule them.

## CLI parity

`docs/CLI.md` is the single-image command Canvas and Lucent call: scale or long side, format, quality, lossless JPEG XL, model tier, JSON lines, exit codes, dry-run, and the suite GPU lock. It does not open a window.

Still ahead:

* A folder and its subfolders, with one JSON event per item.
* Cancel and resume of that batch from the command line, the way the window already restores `queue.json`.

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
