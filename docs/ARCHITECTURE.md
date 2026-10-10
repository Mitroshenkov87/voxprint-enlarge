# Architecture

Voxprint AI Enlarge is the image app. Voxprint AI Clarity is the video app.
They share one engine, owned here and co-maintained by Clarity.

```text
main.py                  window, splash, CLI
enlarge/                 Qt UI, suite.json, app settings
voxprint_engine/         importable by Clarity (no Qt)
  registry/              manifest schema, licence gate, catalog.json
  worker/                JSON-line process, Lanczos stand-in
  gpu_lock.py            %TEMP%/voxprint-gpu.lock
  vram.py                budget = free_now - max(2 GB, 8% total)
  thermal.py             suite pause rule (not wired to the Lanczos path)
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

## Registry

One object per model: `id`, `name`, `arch`, `scale` (1, 2, or 4), `role`
(`upscale`, `restore`, `face`, `generative`), `licence`, `author`,
`attribution`, `source_url`, `sha256` (empty until the weight file is pinned),
`vram_mb`, `tile` (0 lets the worker choose), `filename`, `summary`,
`selectable`.

`validate_manifest(..., release=True)` accepts CC-BY, Apache-2.0, MIT,
BSD-2/3, CC0, and ISC. Non-commercial and missing licences are rejected in
every build. The Credits screen lists models whose licence is CC-BY.

Face and generative entries are in the catalogue so the credits and the gate
exist, and they are not selectable in this build.

## Worker

The app starts `python -m voxprint_engine.worker`. Messages are one JSON object
per line on stdin and stdout. Logs go to stderr.

* `{"op":"ready","engine":"lanczos","pid":N}` when the process is up
* `{"op":"upscale", ...}` with `id`, `src`, `dst`, `scale` (2 or 4), `format`,
  `quality`, `model_id`, `tile`, `model_tile`, `double`, `batch`, `device`
* `{"op":"progress","id","fraction","message"}`
* `{"op":"result", ...}` including `engine`, `tile`, `overlap`,
  `vram_budget_gb`, `spilled`, `next_batch`
* `{"op":"error","id","ok":false,"message"}`
* `{"op":"shutdown"}` stops the process
* `{"op":"ping"}` answers `pong`

The child holds `%TEMP%/voxprint-gpu.lock` for each job (`owner` is
`enlarge`). The lock is stale when the pid is gone or `eta` is more than two
hours old. Timestamps are ISO-8601 UTC.

`next_batch` applies the suite spill rule: halve the batch when `spilled` is
true, and never go below 1. Phase 1 sends one image at a time and does not
spill. Clarity can send `batch` for a frame group and read `next_batch` back.

Video paths are rejected. Video frames, scene cuts, and NVDEC belong in a
later revision of this package, co-maintained with Clarity.

## What this build does not do

Neural inference, ONNX, TensorRT, face restore, and downloading weights.
The Run button and `--upscale` resize with Lanczos so the flow can be tried.
The chosen `model_id` is still sent and returned, so the UI and the worker
already agree on the job.
