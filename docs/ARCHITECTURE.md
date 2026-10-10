# Architecture

Voxprint AI Enlarge is the image app. Voxprint AI Clarity is the video app.
They share one engine, owned here and co-maintained by Clarity.

```text
main.py                  window, splash, CLI
enlarge/                 Qt UI, suite.json, app settings
voxprint_engine/         importable by Clarity (no Qt)
  registry/              manifest schema, licence gate, catalog.json
  worker/                JSON-line process
  infer.py               Lanczos, ONNX, then PyTorch
  session.py             one ONNX provider at a time
  tile.py                overlapping feathered tiles
  providers.py           TensorRT, CUDA after a failed engine build
  hardware.py            compute-capability warning (does not block)
  store.py               shared models folder and .users.json
  paths.py               models dir and TensorRT cache
  gpu_lock.py            %TEMP%/voxprint-gpu.lock
  vram.py                budget = free_now - max(2 GB, 8% total)
  thermal.py             suite pause rule
```

Import from Clarity:

```python
from voxprint_engine import API
from voxprint_engine.registry import combined_catalog, validate_manifest
from voxprint_engine.worker import WorkerClient, UpscaleRequest
from voxprint_engine.vram import vram_budget_gb, choose_tile, note_spill
from voxprint_engine.gpu_lock import gpu_job
from voxprint_engine.thermal import decide
```

`API` is 1. A breaking change to the manifest or the worker messages bumps it.
How a job actually runs (providers, tiles, downloads, the GPU benchmark) is
`docs/ENGINE.md`.

## Registry

One object per model: `id`, `name`, `arch`, `scale` (1, 2, or 4), `role`
(`upscale`, `restore`, `face`, `generative`), `licence`, `author`,
`attribution`, `source_url`, `sha256` (empty until the weight file is pinned),
`vram_mb`, `tile` (0 lets the worker choose), `filename`, `summary`,
`selectable`, and optional `download_url` (https, and only together with a sha256).

`validate_manifest(..., release=True)` accepts CC-BY, Apache-2.0, MIT,
BSD-2/3, CC0, and ISC. Non-commercial and missing licences are rejected in
every build. The Credits screen lists models whose licence is CC-BY.

The three pinned upscalers and Lanczos are selectable. Face, generative, and
the other photo entries stay in the catalogue for credits and the licence gate.
They are not selectable. Pinned hashes, folders, and the licence notes are in
`docs/ENGINE.md`.

## Worker

The app starts `python -m voxprint_engine.worker`. Messages are one JSON object
per line on stdin and stdout. Logs go to stderr.

* `{"op":"ready","engine":"voxprint","pid":N}` when the process is up
* `{"op":"upscale", ...}` with `id`, `src`, `dst`, `scale` (2 or 4), `format`
  (`avif`, `jxl`, `webp`, `jpeg`, or `heic` when a system codec is present),
  `quality`, `lossless` (JPEG XL only), `model_id`, `tile`, `model_tile`,
  `double`, `batch`, `device`, `engine` (`auto`, `onnx`, or `lanczos`),
  `custom_models_dir`
* `{"op":"progress","id","fraction","message"}`
* `{"op":"result", ...}` including `engine`, `tile`, `overlap`,
  `vram_budget_gb`, `spilled`, `next_batch`
* `{"op":"error","id","ok":false,"message"}`
* `{"op":"shutdown"}` stops the process
* `{"op":"ping"}` answers `pong`

The child holds `%TEMP%/voxprint-gpu.lock` for the inference (`owner` is
`enlarge`). A download finishes before the lock is taken. The lock is stale
when the pid is gone or `eta` is more than two hours old. Timestamps are
ISO-8601 UTC.

`next_batch` applies the suite spill rule: halve the batch when `spilled` is
true, and never go below 1. Enlarge still sends one image at a time. Clarity
can send `batch` for a frame group and read `next_batch` back. A tile that
does not fit is halved as well (see `docs/ENGINE.md`).

Video paths are rejected. Video frames, scene cuts, and NVDEC belong in a
later revision of this package, co-maintained with Clarity.

## What this build does not do

Face restore and generative upscale are registered and not run. Video frames,
scene cuts, and NVDEC stay out of this package until Clarity co-maintains that
path. The thermal rule is implemented and is not yet applied between batches.
