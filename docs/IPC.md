# Enlarge hand-off protocol 1

Lucent (and any other suite app) is the client. Enlarge is the server. This file is the stable public surface: the commands below, the discovery records, and the JSON messages. A major change bumps `protocol`. Readers ignore object keys they do not recognise.

The checkout command is `python main.py`. An install also provides `voxprint-enlarge` (and, on Windows, `voxprint-enlarge-cli`) with the same flags. The installer calls `--register` after it copies files.

The window and `python main.py --upscale` still refuse to start without an NVIDIA GPU that CUDA can use on a driver from the 600 series or newer. Two commands are the exception: `--capabilities --json` and `--serve --hidden` start anyway, so a client can see that there is nothing to run.

## Discovery

Enlarge writes these records. Lucent only reads them.

Windows. The installer writes HKLM when it is elevated, and always writes HKCU. A reader still accepts an older HKCU `Exe` for one version.

| Location | Values |
|---|---|
| `HKLM` and `HKCU` `\Software\Voxprint\Enlarge` | `InstallPath`, `Exe` (the CLI), `Version`, `IpcName`, `IpcBase`, `ProtocolVersion` (`1`, `REG_DWORD`) |
| `HKLM` and `HKCU` `\Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-enlarge-cli.exe` | default value is the CLI; `Path` is its directory |
| `HKLM` and `HKCU` `\Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-enlarge.exe` | default value is the window launcher when that file is beside the CLI, otherwise the same exe |

`IpcName` is the pipe name `QLocalSocket` connects to: `voxprint-enlarge-` plus the first 16 hex characters of `sha256` of the user SID string (UTF-8). The full pipe is `\\.\pipe\` plus that name. The pipe ACL allows the current user. If the SID cannot be read, the hash is of the `USERNAME` environment variable.

Linux:

* `$XDG_DATA_HOME/applications/ai.voxprint.Enlarge.desktop` (`~/.local/share` when `XDG_DATA_HOME` is unset). `NoDisplay=true`. `Exec` is the executable plus `--serve --hidden`. `X-Voxprint-Ipc` is the socket path. `X-Voxprint-Ipc-Base` is `voxprint-enlarge`. `X-Voxprint-Protocol=1`. `X-Voxprint-Version` is the version string. `X-Voxprint-Exe` is the absolute executable path. The installed executable is `/opt/voxprint-ai-suite/enlarge/voxprint-enlarge`. The menu entry is `/usr/share/applications/voxprint-enlarge.desktop`, named VOXPrint AI Enlarge.
* An executable `voxprint-enlarge` in `/usr/local/bin`, and the discovery wrapper in `~/.local/bin`.
* For one version, a reader also accepts `$XDG_DATA_HOME/voxprint/enlarge/voxprint-enlarge` when the `/opt` file is absent.

The socket path is `$XDG_RUNTIME_DIR/voxprint/enlarge.sock`. The directory is mode `0700` and the socket is mode `0600`. When `XDG_RUNTIME_DIR` is unset the directory is under the temp folder as `voxprint-run-<uid>`. A Flatpak build, when one exists, is `flatpak run ai.voxprint.Enlarge`.

```bash
python main.py --register [--exe PATH]
```

## Capabilities

```bash
python main.py --capabilities --json
```

Prints one JSON object on stdout and exits 0, including on a machine with no usable GPU. `--capabilities` without `--json` exits 2. `VOXPRINT_CI_LANCZOS` does not change this object: the GPU fields describe the machine.

| Field | Meaning |
|---|---|
| `protocol` | `1` |
| `app` | `enlarge` |
| `version` | The same version `--version` prints. A stamped `build_info.json` wins, then `BUILD.json`, then the source constant. This tree's `BUILD.json` version is `0.2.0-rc.6` |
| `voxprint_home` | The resolved Voxprint home path |
| `voxprint_home_source` | `env` when `VOXPRINT_HOME` is set, otherwise `default` |
| `gpu` | `name`, `cc` (`8.9`), `status` (`ok`, `warn`, or `none`), `reason` (`null` unless `status` is `none`: `no_nvidia` or `driver_too_old`) |
| `models` | List of `{id, preset, quality, scales, installed}` |
| `presets` | `photo`, `art` |
| `max_scale` | `4` |
| `input_formats` | Extensions Enlarge can open, as names (`png`, `jpeg`, `webp`, `avif`, `jxl`, `bmp`, `tiff`) |
| `output_formats` | `avif`, `jxl`, `webp`, `jpeg`, and `heic` only when a system HEIC encoder is present. PNG is never an output |
| `default_output_format` | `avif` |

`gpu.status` is `none` when `nvidia-smi` reports no GPU, or the driver major is below 600. It is `warn` when a GPU is usable and its compute capability is below 8.9 (RTX 40 / Ada). Otherwise it is `ok`. A warning does not block a job. `none` does: the window stays closed and a job returns `gpu_unsupported`.

`preset` today is `photo` or `art`. The enum is open: a later Enlarge can add a preset string, and clients must tolerate preset values they do not know. `quality` is `fast` or `max`. `scales` is `[2, 4]`.

| Preset | Quality | Model id |
|---|---|---|
| `photo` | `fast` | `4x-nomos2-realplksr-dysample` |
| `photo` | `max` | `4x-nomos2-hq-dat2` |
| `art` | `fast` | `realesrgan-x4plus-anime6b` |
| `art` | `max` | `realesrgan-x4plus-anime6b` |

## Transport

`python main.py --serve --hidden` listens and does not open a window until a request sets `open_ui` to `true`. If nothing is listening, the client starts that command and waits up to 3 seconds for the socket or pipe.

Messages are UTF-8 JSON, one object per line, each with `"protocol": 1`. Unknown fields are ignored. A line that is not a JSON object, or whose `protocol` is not `1`, gets `error` with code `internal`.

Jobs run one at a time. Each job holds the suite GPU lock (`owner` `enlarge`) until it finishes or is cancelled. Paths only: the messages carry file paths, never image bytes.

### Client to server

`hello` — any extra fields are ignored. The server answers with its own `hello`.

`enlarge.request`:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Client-chosen job id |
| `input` | yes | Source image path |
| `scale` | yes | `2` or `4` |
| `preset` | with no `model` | `photo` or `art` today. Another string is `bad_input` unless `model` is set |
| `model` | no | Catalog id. When set, it selects the model and `preset` is not used for that choice |
| `quality` | no | `fast` (default) or `max`. This picks the model. The file is still encoded at the app's default quality (60). Lossless is not part of this protocol |
| `format` | no | Output id. Empty means `avif` |
| `output` | no | Exact file path. A collision becomes `<stem>-2<suffix>`, then `-3`, and so on |
| `output_dir` | no | Directory for the default name. Empty means the source's directory |
| `open_ui` | no | JSON `true` opens the window with this image loaded. Any other value leaves the window closed. The window still stays closed when `gpu.status` is `none` |

`cancel` — `{ "op": "cancel", "id": "<job id>" }`. The running job stops. The reply is `error` / `cancelled` / `retryable: false`. A placeholder file is removed.

### Server to client

`hello` — `version`, `protocol_min` (`1`), `protocol_max` (`1`).

`accepted` — `id`, `queue_pos` (how many jobs were already ahead when this one was queued).

`progress` — `id`, `pct` (0–100), `stage`, `eta_s` (a number, or `null` when there is no estimate). Stages are `queued`, `waiting_gpu`, `loading`, `building_engine`, `loading_cached_engine`, `tiles`, `encoding`. `building_engine` is the first TensorRT build, and only when the TensorRT provider stayed active. Its `eta_s` is the time left in a 90 second build, not a linear guess from the first fraction. `loading_cached_engine` is a warm engine cache. Both names are additive; unknown stage strings are ignored. A `waiting_gpu` line also carries `lock_holder` with `owner`, `pid`, and `job` from the suite lock file.

`warning` — `id`, `code`, `message`. The job continues. `code` `gpu_provider_unavailable` means ONNX Runtime did not keep a GPU provider (for example `cublasLt64_13.dll` missing, Error 126) and the run fell back to CPU. That line is not `building_engine`. Canvas's 4K chain into Enlarge reads the same worker, so the same warning applies there.

`done` — `id`, `output` (path), `width`, `height`, `format`, `ms`.

`error` — `id`, `code`, `message`, `retryable`.

| Code | Retryable | When |
|---|---|---|
| `gpu_unsupported` | no | No NVIDIA GPU, or the driver is older than the 600 series |
| `gpu_busy` | yes | The suite GPU lock was not acquired before the job's lock timeout. The error object's `retryable` field is true. A socket client that omits `lock_timeout` still waits 2 seconds. The CLI default is 600 |
| `oom` | no | The worker ran out of memory |
| `bad_input` | no | Missing file, scale not 2 or 4, quality not `fast` or `max`, unknown preset with no model, or a parent-lock pid that is not the lock file's pid |
| `unsupported_format` | no | The source type or the output format is not supported. PNG is not an output |
| `io_error` | no | The output path could not be reserved or written |
| `cancelled` | no | The client cancelled the job, or disconnected |
| `model_missing` | yes | The model id is absent, not selectable, or the pinned weights are not on disk. A headless job does not download them |
| `internal` | no | Anything else, including a bad message |

### Output names

The default path is `<name>.enlarged-<scale>x.<ext>` next to the source (or inside `output_dir`). `<ext>` is `.avif`, `.jxl`, `.webp`, `.jpg`, or `.heic`. If that file exists, the next candidate is `<name>.enlarged-<scale>x-2.<ext>`, then `-3`, and so on. The server reserves the name with an exclusive create and never overwrites an existing file. A job that fails or is cancelled deletes the placeholder it reserved.

Messages carry both `op` and `type` set to the same name. Readers that already match `op` keep working. `type` is the name in `docs/CLI.md`.

## CLI

The command other apps call, including exit codes, `--long-side`, `--dry-run`, the parent GPU lock, and warm-server forwarding, is `docs/CLI.md`. `--json-progress` is required. The process does not open a window. `VOXPRINT_CI_LANCZOS=1` is for tests. It is not a user mode, and it does not change `--capabilities`.

```bash
python main.py --input FILE --scale 2|4 --preset photo|art \
  [--model ID] [--quality fast|max] [--format avif|jxl|webp|jpeg|heic] \
  [--output PATH] --json-progress
```
