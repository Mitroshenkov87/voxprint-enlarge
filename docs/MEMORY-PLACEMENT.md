# Memory placement

Design for the next `voxprint_engine`. This document does not add code, and it does not change `API`. `API` stays 1. A later pull request can add the names below to `voxprint_engine.api.__all__` and to the contract test in the same change.

The goal is to run a larger model on a 16 GB or 24 GB card by placing each piece in graphics memory, pinned system memory, or disk, instead of loading the whole model onto the card at once.

## What the engine does today

The VRAM budget is already fixed:

```text
budget = free_now - max(2 GB, 8% of total)
```

`voxprint_engine.vram` measures that before a batch. GB is 1024 cubed. `headroom_gb` is the part left unused. `auto_tile` then picks 128, 256, 512, or 768. `choose_tile` lets an explicit user tile win, otherwise it clamps the model's tile hint to that ladder. `note_spill` halves the batch and never goes below 1. `halve_tile` shrinks a tile that does not fit. `is_memory_error` recognises an allocator failure.

The suite GPU lock is one file. A parent that already holds it passes its pid. The worker shares that lock and does not delete it. A job that takes the lock itself releases it when the job ends.

TensorRT engines are cached under `VOXPRINT_TRT_CACHE`, or `%LOCALAPPDATA%\Voxprint\trt-cache`, or `~/.cache/voxprint/trt-cache`. A session uses one ONNX provider. TensorRT is first. CUDA is used only when that engine does not build, or when the TensorRT provider is absent.

The registry manifest already carries `vram_mb` and `tile`. Those are hints. `role` can be `upscale`, `restore`, `face`, or `generative`. Face and generative rows are not selectable in this build. Unknown manifest fields are ignored, which is how a later field can arrive without a breaking change.

## Tiers

A tier is the graphics memory a model is allowed to occupy after the budget rule, not the card's sticker size.

| Tier | Card the planner expects | Budget left for weights and activations |
|---|---|---|
| 16 | About 16 GB | `vram_budget_gb` on that card |
| 24 | About 24 GB or more | `vram_budget_gb` on that card |

The planner reads the live budget. It does not hard-code 16 or 24 as a second rule. The tier name is only how a model declares what it was measured on.

Model Forge records those measurements. They enter this engine as optional manifest fields, for example `tier_gb` and a per-component size list. A build that does not know a field skips it. A model with no tier fields keeps today's behaviour: one ONNX session, one tile ladder, no staged unload.

## Allocation planner

The planner takes the budget, the model's component list, and the live free memory. It returns a placement for each component:

* `vram` when the component and its activations fit in the budget together with the tile.
* `pinned` when the component should sit in pinned system RAM and be copied to the card for the step that needs it.
* `disk` when it should stay in the TensorRT cache or the weights file until that step.

Small components and the image encoders stay on CPU RAM. AVIF, JPEG XL, and WebP encoding already run off the card. The planner must not move them into VRAM to "make room"; they never needed the card.

Components load and unload in order. The resident set is the component in use plus the next one being prefetched. When a component finishes, its device memory is released before the next large component is staged, unless the budget still holds both.

## Prefetch

While the GPU computes a tile, the host can read the next weight block or the next tile from disk or pinned RAM over PCIe. The prefetch buffer is pinned host memory so the copy does not stall on a page fault. One buffer is in flight. A second buffer waits until the first copy completes. The compute stream and the copy stream stay separate so the overlap is real.

Pinned buffers are allocated up front from system RAM, sized from the largest component the plan marked `pinned` or `disk`. If that allocation fails, the planner drops prefetch and loads each component only when the GPU is idle. The job still runs. It is slower.

## GPU lock and TensorRT

The planner runs inside the same suite lock as the job. Prefetch does not take a second lock. A parent lock is still shared: placement does not release it and does not replace the pid check.

TensorRT engines stay in the existing cache. A component whose engine is already built is `disk` until the session needs it, then `vram` for that step. Rebuilding an engine is still the slow path described in `docs/ENGINE.md`. Placement does not change the provider order: TensorRT first, CUDA only when that engine does not build.

## API sketch

These names are not exported. A later change adds them together, still under `API` 1:

```text
MemoryTier          measured tier from the manifest, or unknown
Placement           vram, pinned, or disk
ComponentPlan       one component, its bytes, and its placement
plan_placement(budget_gb, manifest) -> list[ComponentPlan]
resident_bytes(plan) -> int
```

`plan_placement` is a pure function of the budget and the manifest. It does not open a GPU context. `vram_budget_gb`, `choose_tile`, `halve_tile`, and `note_spill` stay as they are. The image `upscale` path keeps today's defaults until a caller opts into a plan.

## Tests and benchmarks

Unit tests stay GPU-free. They feed a fake budget and a manifest with tier fields and assert the placement list, including the case where the budget fits only one large component at a time, the case with no tier fields, and the case where pinned RAM is refused and prefetch is off.

A machine with a GPU records timings with `tools/benchmark_gpu.py`, as `docs/TESTING-GPU.md` already describes. The benchmark compares a staged plan with today's single session on the same image and the same model. The docs change lands with the implementation, not before it.
