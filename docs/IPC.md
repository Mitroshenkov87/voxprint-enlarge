# Enlarge hand-off protocol 1

Lucent (and any other suite app) is the client. Enlarge is the server. This file is the stable public surface: the commands below, the discovery records, and the JSON messages. A major change bumps `protocol`. Readers ignore object keys they do not recognise.

The checkout command is `python main.py`. An install that has called the discovery helper (below) also provides `voxprint-enlarge` with the same flags. There is no installer in this build. The helper only writes the records an installer will call later.

The window and `python main.py --upscale` still refuse to start without an NVIDIA GPU that CUDA can use on a driver from the 600 series or newer. Two commands are the exception: `--capabilities --json` and `--serve --hidden` start anyway, so a client can see that there is nothing to run.

## Discovery

Enlarge writes these records. Lucent only reads them.

Windows, per user:

| Location | Values |
|---|---|
| `HKCU\Software\Voxprint\Enlarge` | `InstallPath`, `Exe`, `Version`, `IpcName`, `ProtocolVersion` (`1`), all `REG_SZ` |
| `HKCU\Software\Microsoft\Windows\CurrentVersion\App Paths\voxprint-enlarge.exe` | default value is `Exe`; `Path` is `InstallPath` |

`IpcName` is the pipe name `QLocalSocket` connects to: `voxprint-enlarge-` plus the first 16 hex characters of `sha256` of the user SID string (UTF-8). The full pipe is `\\.\pipe\` plus that name. The pipe ACL allows the current user. If the SID cannot be read, the hash is of the `USERNAME` environment variable.

Linux:

* `$XDG_DATA_HOME/applications/ai.voxprint.Enlarge.desktop` (`~/.local/share` when `XDG_DATA_HOME` is unset). `NoDisplay=true`. `Exec` is the executable plus `--serve --hidden`. `X-Voxprint-Ipc` is the socket path. `X-Voxprint-Protocol=1`.
* An executable `voxprint-enlarge` in `~/.local/bin`, which is expected to be on `PATH`.

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
| `version` | Enlarge version string, currently `0.1.0-pre` |
| `gpu.status` | `ok`, `warn`, or `none` |
| `gpu.reason` | Present only when `status` is `none`: `no_nvidia` or `driver_too_old` |
| `models` | List of `{id, preset, quality, scales}` |
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

`progress` — `id`, `pct` (0–100), `stage`. Stages are `queued`, `waiting_gpu`, `loading`, `tiles`, `encoding`. `eta_s` is optional and this server does not send it.

`done` — `id`, `output` (path), `width`, `height`, `format`, `ms`.

`error` — `id`, `code`, `message`, `retryable`.

| Code | Retryable | When |
|---|---|---|
| `gpu_unsupported` | no | No NVIDIA GPU, or the driver is older than the 600 series |
| `gpu_busy` | yes | The suite GPU lock is still held after 2 seconds |
| `oom` | yes | The worker ran out of memory |
| `bad_input` | no | Missing file, scale not 2 or 4, quality not `fast` or `max`, unknown preset with no model |
| `unsupported_format` | no | The source type or the output format is not supported. PNG is not an output |
| `io_error` | yes | The output path could not be reserved or written |
| `cancelled` | no | The client cancelled the job, or disconnected |
| `model_missing` | no | The model id is absent or not selectable |
| `internal` | yes | Anything else, including a bad message |

### Output names

The default path is `<name>.enlarged-<scale>x.<ext>` next to the source (or inside `output_dir`). `<ext>` is `.avif`, `.jxl`, `.webp`, `.jpg`, or `.heic`. If that file exists, the next candidate is `<name>.enlarged-<scale>x-2.<ext>`, then `-3`, and so on. The server reserves the name with an exclusive create and never overwrites an existing file. A job that fails or is cancelled deletes the placeholder it reserved.

## CLI fallback

The same `accepted`, `progress`, `done`, and `error` objects are written to stdout, one per line. Exit 0 after `done`. Exit 1 after `error`. Exit 2 when the arguments are incomplete. `--json-progress` is required.

```bash
python main.py --input FILE --scale 2|4 --preset photo|art \
  [--model ID] [--quality fast|max] [--format avif|jxl|webp|jpeg|heic] \
  [--output PATH] --json-progress
```

`open_ui` is not available on this command. The job still holds the GPU lock. On a machine with no usable GPU the stream ends with `gpu_unsupported` and the process exits 1, unless the internal CI flag `VOXPRINT_CI_LANCZOS=1` is set. That flag is for tests. It is not a user mode, and it does not change `--capabilities`.
