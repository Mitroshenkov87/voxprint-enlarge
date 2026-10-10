# Voxprint AI Enlarge: decisions

2026-10-10:

* Name: Voxprint AI Enlarge. Clean-room app. No Upscayl code and no non-commercial models.
* UI matches the Voxprint suite (acrylic / Mica, shared theme and components).
* Images only. No video features. Video belongs to Voxprint AI Clarity.
* Enlarge owns the shared inference engine and model registry. Clarity co-maintains them.
* Models are licence-safe only (CC-BY, Apache, MIT, BSD). CC-BY models are credited in the app.
* Platforms: Windows 11 24H2+ and Linux releases from 2025 onward. Local only.
* Forward-only. The GPU floor is NVIDIA RTX 40 (Ada, compute capability 8.9) and newer. A card below 8.9 raises a diagnostics warning and the job still runs. DirectML is not a provider. CUDA EP is used when a TensorRT engine does not build. Lanczos stays the no-GPU preview.
* The x86-64 CPU baseline is x86-64-v3, without an AVX-512 path. The same Python code runs on ARM.
* VRAM use follows the suite margin: free memory minus max(2 GB, 8% of the card).
* The program is Apache-2.0, the same licence as the other Voxprint apps.
* Output files default to AVIF. The main set is AVIF, JPEG XL (lossy or lossless), and WebP. JPEG is a legacy option. HEIC is a legacy option only when a system codec is present, and no HEIC encoder is bundled. PNG is an input format, not an output. JPEG and HEIC are never the default.
* The repository stays private until the first release. Only the owner changes visibility.
