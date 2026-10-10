# Voxprint AI Enlarge: research notes

Status of this file: background for the registry. It is not a promise that a
model is downloaded or running. Weights are not in the repository.

The app is a clean-room image upscaler. Workflow ideas worth keeping: pick
files or a folder, pick a model, pick scale and format, run; a before/after
slider with zoom; a batch queue; double upscale; a custom models folder;
GPU and tile settings; a credits screen for CC-BY. Do not copy Upscayl code
or its non-commercial models.

## Models in the registry

Licence-safe entries, checked against the project's pages on 2026-10-10:

| id | licence | author | source |
|---|---|---|---|
| 4x-nomos2-realplksr-dysample | CC-BY-4.0 | Philip Hofmann | OpenModelDB and the Phhofm/models release. Some cards say "CC-BY-0.4"; the Hugging Face metadata and the OpenModelDB rights block say CC-BY-4.0. |
| 4x-nomos2-hq-dat2 | CC-BY-4.0 | Philip Hofmann | OpenModelDB |
| 4x-nomosuni-span-multijpg | CC-BY-4.0 | Philip Hofmann | OpenModelDB and the GitHub release (CC BY 4.0) |
| 1x-dejpg-realplksr-otf | CC-BY-4.0 | Philip Hofmann | Phhofm/models release `1xDeJPG_realplksr_otf` |
| 1x-denoise-realplksr-otf | CC-BY-4.0 | Philip Hofmann | OpenModelDB |
| realesrgan-x4plus | BSD-3-Clause | Xintao Wang | github.com/xinntao/Real-ESRGAN |
| realesrgan-x4plus-anime6b | BSD-3-Clause | Xintao Wang | same project |
| gfpgan-v1-4 | Apache-2.0 | Xintao Wang, Tencent ARC | github.com/TencentARC/GFPGAN (the licence carves out some third-party components) |
| restoreformer-plus | Apache-2.0 | Zhou et al. | github.com/wzhouxiff/RestoreFormerPlusPlus |
| seedvr2-3b | Apache-2.0 | ByteDance Seed | github.com/ByteDance-Seed/SeedVR |

`sha256` stays empty until a weight file is pinned. Face and generative models
are registered and not selectable in phase 1.

Not in the registry, and not accepted if someone drops them in the custom
folder under a non-commercial or missing licence: Kim2091 models published as
non-commercial, SUPIR, HYPIR, CodeFormer, StableSR, InvSR, LucidFlux, PiSA-SR,
and Upscayl's UltraSharp / Remacri (CC-BY-NC-SA).

## Runtime, later phases

ONNX Runtime, with TensorRT EP preferred over CUDA EP. DirectML is a fallback
only. Windows ML on 24H2 is in scope later. The path is spandrel's MIT core
(not `spandrel_extra_arches`) from `.pth` to ONNX fp16, with TensorRT engines
cached under `<Voxprint home>/trt-cache/<gpu>-<driver>-<trt>/`. Tiles overlap
by the feather reported today (`overlap_for`) and are sized from the VRAM
budget. Inference stays in the worker process.

VRAM budget, measured again before each batch: `free_now - max(2 GB, 8% of total)`.
If a batch spills into system memory, the next batch is half the size (at least 1).

Thermal rule, implemented in `voxprint_engine.thermal` and not yet applied to
the Lanczos preview: full speed for 2.5 hours, then pause between batches while
the 5-minute GPU median is at least 83 °C or the GPU has been throttling for
more than 60 seconds. Resume at or below 75 °C.
