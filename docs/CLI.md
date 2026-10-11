# Enlarge CLI

This is the stable command other Voxprint apps call. Canvas and Lucent use this instead of the socket. The process never opens a window.

A warm `voxprint-enlarge --serve` for this user receives the job over protocol 1. If nothing is listening, the same process runs the job. Stdout is the same JSON lines either way. Human text goes to stderr.

Every JSON object has `"protocol": 1`. Unknown fields are ignored.

## Invocation

```text
voxprint-enlarge --input <file> --output <file> (--scale 2|4 | --long-side <px>) [--format avif|jxl|webp|jpeg|heic] [--lossless] [--quality 1-100] [--preset photo|art] [--model <id>] [--quality-tier fast|max] [--lock-timeout <s>] [--parent-lock-pid <pid>] [--dry-run] --json-progress
voxprint-enlarge upscale ...
```

`upscale` is a subcommand alias with the same flags. `--json-progress` is required. Without it the process prints usage on stderr and exits 2.

The flag list and the defaults below are generated from `enlarge/cli_parser.py` when the docs site is built. That page is the [CLI reference](https://mitroshenkov87.github.io/voxprint-enlarge/generated/cli-reference/). This file stays the contract.

| Flag | Meaning |
|---|---|
| `--input` | Source image. Required |
| `--output` | Destination file. Optional. `--out` is the same flag |
| `--scale` | `2` or `4`. Omit both scale and long-side and the scale is 4 |
| `--long-side` | Target long side in pixels. Not together with `--scale` |
| `--format` | `avif` (default), `jxl`, `webp`, `jpeg`, or `heic` when a system HEIC encoder is present. Never PNG |
| `--lossless` | JPEG XL only |
| `--quality` | Encode quality `1`–`100`. Default 60. The words `fast` and `max` still select the model tier, which is the flag this command already had |
| `--quality-tier` | Model tier `fast` (default) or `max`. This wins over `--quality fast` or `--quality max` |
| `--preset` | `photo` (default) or `art`. An explicit `--model` overrides it |
| `--model` | Registry id. It overrides preset and tier |
| `--lock-timeout` | Seconds to wait for the suite GPU lock. Default 600 |
| `--parent-lock-pid` | Pid that already holds the suite GPU lock. The environment variable `VOXPRINT_GPU_PARENT_PID` is the same value. The flag wins when both are set |
| `--dry-run` | Validate and print the plan. No model load, no GPU, no output file |
| `--json-progress` | Required. Stdout is JSON lines |

The older `voxprint-enlarge --upscale FILE --out FILE` command is unchanged. It prints a path or an error for a person, it still refuses to start without a usable NVIDIA GPU, and its exit code is 0 or 1.

## Long side

`--long-side N` reads the source size, picks the smallest model scale in `{2, 4}` whose long side is at least N, runs that scale, then resizes so the long side is exactly N. The short side is `round(short * N / long)` (Python round-half-even) and is at least 1. The output name uses that model scale, so a 20 px long side on an 8 px image is `enlarged-4x` because 2×8 is 16 and 4×8 is 32.

If `4 * max(width, height) < N`, the result is `bad_input` and the process exits 2. Passing `--scale` and `--long-side` together is also `bad_input`.

## Output name

With no `--output`, the file is `<name>.enlarged-<scale>x.<ext>` next to the input. `<ext>` is `avif`, `jxl`, `webp`, `jpg`, or `heic`. If that file exists, the next name is `<name>.enlarged-<scale>x-2.<ext>`, then `-3`, and so on. An explicit `--output` that already exists becomes `<stem>-2<suffix>`, then `-3`. Nothing is overwritten. A failed or cancelled job deletes the file it reserved.

## Stdout

One JSON object per line. These are the IPC-ENLARGE shapes. `type` is the name. `op` is the same string, so a reader from the published socket still matches.

```json
{"protocol":1,"type":"accepted","id":"cli","queue_pos":0}
{"protocol":1,"type":"progress","id":"cli","pct":1.0,"stage":"waiting_gpu","eta_s":12.0,"lock_holder":{"owner":"canvas","pid":1001,"job":"canvas:export"}}
{"protocol":1,"type":"progress","id":"cli","pct":42.0,"stage":"tiles","eta_s":8.0}
{"protocol":1,"type":"done","id":"cli","output":"/absolute/or/given/path","width":20,"height":15,"format":"webp","ms":40.0}
{"protocol":1,"type":"error","id":"cli","code":"bad_input","message":"...","retryable":false}
```

`stage` is `queued`, `waiting_gpu`, `loading`, `building_engine`, `loading_cached_engine`, `tiles`, or `encoding`. The two engine stages are additive: a reader that does not know them can ignore the string. `building_engine` is the first TensorRT build and reports `eta_s` as the time left in a 90 second build. `loading_cached_engine` is a warm cache. `eta_s` is a number, or `null` when there is no estimate yet. `lock_holder` is present on `waiting_gpu` and is read from the suite lock file: `owner`, `pid`, `job`.

An argument error on a job that passed `--json-progress` is one JSON object on stdout, `code` `bad_input`, and the process exits 2. The usage text stays on stderr when `--json-progress` was not passed.

| Code | Retryable | Exit |
|---|---|---|
| `internal` | no | 1 |
| `bad_input` | no | 2 |
| `unsupported_format` | no | 3 |
| `gpu_unsupported` | no | 4 |
| `gpu_busy` | yes | 5 |
| `oom` | no | 6 |
| `model_missing` | yes | 7 |
| `io_error` | no | 8 |
| `cancelled` | no | 130 |

Exit 0 is `done`. Exit 2 is also bad arguments, before a job id exists. `model_missing` means the id is unknown or the pinned weights are not on disk. A headless job does not download them. SIGINT and SIGTERM cancel the job, delete the partial file, and exit 130.

## GPU lock

With no parent pid, Enlarge takes the suite lock for the job and releases it when the job ends. The wait is `--lock-timeout` seconds (default 600). On timeout the code is `gpu_busy`, exit 5, and the JSON field `retryable` is true.

When the caller already holds the lock, it passes `--parent-lock-pid` or `VOXPRINT_GPU_PARENT_PID`. Enlarge reads the lock file and continues only when that file's `pid` is the one given. The owner name is not checked. The job then runs inside the lock: it does not wait, and it does not delete the file. A missing file or a different pid is `bad_input`, exit 2.

## Capabilities

```text
voxprint-enlarge --capabilities --json
```

Prints one JSON object and exits 0. It does not need a GPU, a network, or a window. If a warm server is listening, the object comes from that server. Otherwise it is built in this process. `--capabilities` without `--json` exits 2.

```json
{
  "protocol": 1,
  "app": "enlarge",
  "version": "0.2.0-rc.3",
  "gpu": {"name": null, "cc": null, "status": "none", "reason": "no_nvidia"},
  "models": [
    {"id": "4x-nomos2-realplksr-dysample", "preset": "photo", "quality": "fast", "scales": [2, 4], "installed": false}
  ],
  "presets": ["photo", "art"],
  "max_scale": 4,
  "input_formats": ["avif", "bmp", "jpeg", "jxl", "png", "tiff", "webp"],
  "output_formats": ["avif", "jxl", "webp", "jpeg"],
  "default_output_format": "avif"
}
```

`gpu.status` is `ok`, `warn`, or `none`. `gpu.reason` is `null` unless `status` is `none`, where it is `no_nvidia` or `driver_too_old`. `installed` is true only when the pinned weight file is on disk. Model ids come from the registry. `VOXPRINT_CI_LANCZOS` does not change this object.

## Dry run

`--dry-run` checks the arguments, the source, the format, the model id, whether the weights are on disk, and the GPU status. It prints the JSON it would print for that outcome. On success that is `accepted`, then `done` with the planned path and the planned width and height, and `ms` is 0. It does not create the file, load a model, or open a GPU context, so it runs on a GPU-less machine. Argument, format, model-missing, and GPU failures use the same exit codes as a real run. It does not forward to a warm server.

## Discovery

The executable other apps start is the one the installer wrote.

| Platform | Path |
|---|---|
| Windows | `%LOCALAPPDATA%\Voxprint\Enlarge\VoxprintEnlarge.exe` |
| Linux | `$XDG_DATA_HOME/voxprint/enlarge/voxprint-enlarge` (`~/.local/share` when `XDG_DATA_HOME` is unset) |

`voxprint-enlarge --register` records that absolute path in `Exe` and in `X-Voxprint-Exe`. The base name from APP-DISCOVERY is `voxprint-enlarge` (`IpcBase`, `X-Voxprint-Ipc-Base`). `IpcName` and `X-Voxprint-Ipc` stay the listen name the published RC already stored, so a socket client keeps working. Linux also writes `X-Voxprint-Version`. Windows `ProtocolVersion` is a `REG_DWORD`. The per-user App Paths key remains `voxprint-enlarge.exe`.

Fallback when the known file is absent: `voxprint-enlarge` on `PATH`, then the current process.
