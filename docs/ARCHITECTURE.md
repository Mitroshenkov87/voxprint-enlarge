# Architecture

VOXPrint AI Enlarge is the image app. VOXPrint AI Clarity is the video app.
They share one engine, owned here and co-maintained by Clarity.

```text
main.py                  process entry. Qt loads only for the window and --serve
enlarge/cli.py           headless contract (docs/CLI.md)
enlarge/cli_parser.py    argparse source for the process and the generated reference
enlarge/                 Qt UI, suite.json, app settings
enlarge/ipc/             protocol 1 server, jobs, and a Qt-free client (docs/IPC.md)
voxprint_engine/         importable by Clarity (no Qt)
  registry/              manifest schema, licence gate, catalog.json
  worker/                JSON-line process
  infer.py               ONNX, then PyTorch; Lanczos only for CI
  session.py             one ONNX provider at a time
  tile.py                overlapping feathered tiles
  video.py               temporal windows, per-frame tiles, NCHW batch
  providers.py           TensorRT, CUDA after a failed engine build
  store.py               shared models folder and .users.json
  vram.py                budget = free_now - max(2 GB, 8% total)
enlarge/suite_runtime.py shared Python, CUDA wheels, default model; refs go through voxprint_common
voxprint_common          paths, GPU lock, thermal rule, GPU probe, licence, manifest, splash
```

Import from Clarity:

```python
from voxprint_engine import API
from voxprint_engine.registry import combined_catalog, validate_manifest
from voxprint_engine.worker import WorkerClient, UpscaleRequest
from voxprint_engine.vram import vram_budget_gb, choose_tile, note_spill
from voxprint_common.gpu_lock import gpu_job
from voxprint_common.thermal import decide
from voxprint_engine.video import temporal_windows, restore_frames
```

`API` is 1. A breaking change to the manifest or the worker messages bumps it.
How a job actually runs (providers, tiles, downloads, the GPU benchmark) is
`docs/ENGINE.md`. A later placement of weights across graphics memory, pinned
system RAM, and disk is designed in `docs/MEMORY-PLACEMENT.md` and is not
implemented.

## App

`main.py` parses arguments with `enlarge.cli_parser` and then dispatches.
`--version`, `--capabilities`, `--dry-run`, and a headless `upscale` never
import PySide6. The window and `--serve` import Qt inside those functions, so
a machine without libEGL can still answer `--capabilities`.

The product version is one value. A stamped `build_info.json` wins, then the
`version` field of `BUILD.json`, then the source constant. `--version` and
`--capabilities` both read it. The installer `AppVersion` is that same
`BUILD.json` field.

## IPC and CLI

Protocol 1 is `docs/IPC.md`. The headless command is `docs/CLI.md`. The flag
list and the defaults are the argparse parser. The docs site renders that
parser at build time, so the reference page and the process stay the same.
`docs/CLI.md` stays the contract prose.

A warm `--serve` process receives a job over the per-user socket (Linux) or
named pipe (Windows). The client is the stdlib, not Qt. If nothing is
listening, the same process runs the job. `--dry-run` never forwards.
`gpu_busy` is `retryable: true`. So is `model_missing`.

The suite GPU lock is held by the job. A caller that already holds it passes
`--parent-lock-pid`. The worker shares that lock and does not delete the file.

## Packaging

Release builds use a relocatable CPython 3.14 from python-build-standalone.
Weights are not in the installer or the tarball.

| Platform | Install path |
|---|---|
| Windows | `%ProgramFiles%\VOXPrint AI Suite\Enlarge\` (`voxprint-enlarge.exe`, `voxprint-enlarge-cli.exe`) |
| Linux | `/opt/voxprint-ai-suite/enlarge/voxprint-enlarge` |

`BUILD.json` holds `version`, the build-number `offset`, and the codename.
`voxprint_common.buildinfo` adds `GITHUB_RUN_NUMBER` to the offset. Without that
variable, a missing recorded build stays null and the label says `dev`, not
build 0. The installer
runs `--install-suite`, which copies Python 3.14 into the shared suite folder,
installs torch plus the CUDA 13 ONNX Runtime and TensorRT wheels, and downloads
the default model. `--uninstall-suite` drops this app's hold and deletes a
shared file only when no other app still lists it. Every real launch and
`--selftest` re-check that install and repair it. `--install-gpu` still
installs the wheels by themselves. A frozen app refuses that command.

## Registry

One object per model: `id`, `name`, `arch`, `scale` (1, 2, or 4), `role`
(`upscale`, `restore`, `face`, `generative`), `licence`, `author`,
`attribution`, `source_url`, `sha256` (empty until the weight file is pinned),
`vram_mb`, `tile` (0 lets the worker choose), `filename`, `summary`,
`selectable`, and optional `download_url` (https, and only together with a sha256).

`validate_manifest(..., release=True)` accepts CC-BY, Apache-2.0, MIT,
BSD-2/3, CC0, and ISC. Non-commercial and missing licences are rejected in
every build. The Credits screen lists models whose licence is CC-BY.

The three pinned upscalers are selectable. Lanczos stays in the catalogue for
the CI stand-in and is not selectable. Face, generative, and the other photo
entries stay for credits and the licence gate. They are not selectable. Pinned
hashes, folders, and the licence notes are in `docs/ENGINE.md`.

## Worker

The app starts `python -m voxprint_engine.worker`. Messages are one JSON object
per line on stdin and stdout. Logs go to stderr.

* `{"op":"ready","engine":"voxprint","pid":N}` when the process is up
* `{"op":"restore_batch", ...}` restores an NHWC float `.npy` for Clarity.
  The fields and the keep rules are the Video API in `docs/ENGINE.md`.
  The image `upscale` path does not use this op.
* `{"op":"upscale", ...}` with `id`, `src`, `dst`, `scale` (2 or 4), `format`
  (`avif`, `jxl`, `webp`, `jpeg`, or `heic` when a system codec is present),
  `quality`, `lossless` (JPEG XL only), `model_id`, `tile`, `model_tile`,
  `double`, `batch`, `device`, `engine` (`auto` or `onnx`),
  `custom_models_dir`
* `{"op":"progress","id","fraction","message"}`
* `{"op":"result", ...}` including `engine`, `tile`, `overlap`,
  `vram_budget_gb`, `spilled`, `next_batch`
* `{"op":"error","id","ok":false,"message"}`
* `{"op":"shutdown"}` stops the process
* `{"op":"ping"}` answers `pong`

The worker holds `%TEMP%/voxprint-gpu.lock` for the inference (`owner` is
`enlarge`). A hand-off job holds that same lock in the server process; the
worker treats a lock whose pid is its parent as shared and does not take a
second one or release the parent's. A download finishes before the lock is
taken. The lock is stale when the pid is gone or `eta` is more than two hours
old. Timestamps are ISO-8601 UTC.

`next_batch` applies the suite spill rule: halve the batch when `spilled` is
true, and never go below 1. Enlarge still sends one image at a time. Clarity
can send `batch` for a frame group and read `next_batch` back. A tile that
does not fit is halved as well (see `docs/ENGINE.md`).

The image upscale path rejects video files. Clarity sends frames through
`restore_batch`. Scene cuts and NVDEC stay out of this package.

## What this build does not do

Face restore and generative upscale are registered and not run. Scene cuts
and NVDEC stay out of this package. Video frame restore for Clarity is the
Video API in `docs/ENGINE.md`. The thermal rule is implemented and is not
yet applied between batches.
