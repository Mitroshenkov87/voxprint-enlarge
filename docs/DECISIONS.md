# VOXPrint AI Enlarge: decisions

2026-10-10:

* Name: VOXPrint AI Enlarge. Clean-room app. No Upscayl code and no non-commercial models.
* UI matches the VOXPrint suite (acrylic / Mica, shared theme and components).
* Images only. No video features. Video belongs to VOXPrint AI Clarity.
* Enlarge owns the shared inference engine and model registry. Clarity co-maintains them.
* Models are licence-safe only (CC-BY, Apache, MIT, BSD). CC-BY models are credited in the app.
* Platforms: Windows 11 24H2+ and Linux releases from 2025 onward. Local only.
* Forward-only. Enlarge starts only when an NVIDIA GPU is usable through CUDA and the driver is from the 600 series or newer. A card below compute capability 8.9 (RTX 40 / Ada) raises a diagnostics warning and the app still starts. DirectML is not a provider. CUDA EP is used when a TensorRT engine does not build.
* There is no CPU-only mode. Lanczos resampling runs only when `VOXPRINT_CI_LANCZOS=1`, which tests and CI set so they stay offline and GPU-free.
* Python 3.14. Current PyTorch CUDA 13, onnxruntime-gpu, TensorRT, NumPy, Pillow, PySide6, and imagecodecs wheels support 3.14. Python 3.15 has no onnxruntime-gpu wheel. `imagecodecs` is 2026.10.10.
* GPU packages target CUDA 13 (PyTorch `cu130`, `onnxruntime-gpu`, `tensorrt-cu13`). The `cu126` index and the R560 driver branch are not used.
* The x86-64 CPU baseline is x86-64-v3, without an AVX-512 path. The same Python code runs on ARM.
* VRAM use follows the suite margin: free memory minus max(2 GB, 8% of the card).
* The program is Apache-2.0, the same licence as the other VOXPrint apps.
* Output files default to AVIF. The main set is AVIF, JPEG XL (lossy or lossless), and WebP. JPEG is a legacy option. HEIC is a legacy option only when a system codec is present, and no HEIC encoder is bundled. PNG is an input format, not an output. JPEG and HEIC are never the default.
* The repository stays private until the first release. Only the owner changes visibility.
