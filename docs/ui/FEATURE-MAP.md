# Enlarge feature map

One row for each old-window feature and each CLI feature. The old window in
`enlarge/ui/window.py` stays in the tree and is no longer opened. The new
window is `surface_for("enlarge")` on `SimpleLayout`.

| Feature | Where it was | Where it lives now | Status |
| --- | --- | --- | --- |
| Add images | Images step | Drop area. Open is handled, and this surface does not emit it | mapped |
| Add folder | Images step | Not on the built-in file drop. Subfolders stays a preset | mapped |
| Remove, Clear | Images step | Not on the built-in file drop | mapped |
| Include subfolders | Images step | Preset Subfolders, default on. Stored only | mapped |
| Photographs / highest detail / Drawings | Model step, three names | Preset Pictures are. Values are registry ids in `mdl-map.json` | mapped |
| Manage models | Model step | Overflow `models.manage` | mapped |
| Check hash | Model step | Expert is the licence line. The hash check has no shell command | mapped |
| Remove weights | Model step | Overflow Manage models | mapped |
| Scale 2 and 4 | Output step | Preset Size. Default is four times as big | mapped |
| Double upscale | Output step and `--double` | Expert Double upscale. The panel shows the resulting factor | mapped |
| AVIF, JPEG XL, WebP | Output step | Preset File type. Default AVIF | mapped |
| JPEG, HEIC | Output step | Expert file type | mapped |
| Lossless JPEG XL | Output step and `--lossless` | Preset Lossless. The job refuses it for other file types | mapped |
| Quality | Output step and `--quality` | Expert. Picture presets write 60 | mapped |
| Output folder, beside the source | Output step and `--output` / `--output-dir` | Expert Next to the picture and Output folder | mapped |
| Tile size | Output step and Settings | Expert. Automatic is the default | mapped |
| GPU | Settings | Expert. The same suite value | mapped |
| Custom models folder | Settings | Expert | mapped |
| Shared models folder | Settings | Expert. Writes the suite folder | mapped |
| Model id and licence | Model step detail line | Expert Model and Licence | mapped |
| Queue | Run step | History strip | mapped |
| Compare, before / after, zoom | Run step | The built-in enlarge result is a picture, not a wipe | mapped |
| Run | Run step | Primary action Enlarge | mapped |
| Cancel | Run step | The shell Cancel control while a job is busy | mapped |
| GPU warning | Output step | A sentence when the probe says warn | mapped |
| Settings dialog | Window header | Expert panel. One settings state | mapped |
| Credits | Window header | No overflow command on this surface | mapped |
| Splash | Shared splash | Unchanged `voxprint_common.ui.splash` | mapped |
| `--input` | CLI | unchanged CLI | CLI-only |
| `--output` / `--out` | CLI | unchanged CLI | CLI-only |
| `--output-dir` | CLI | unchanged CLI | CLI-only |
| `--scale` | CLI | unchanged CLI | CLI-only |
| `--long-side` | CLI | unchanged CLI | CLI-only |
| `--format` | CLI | unchanged CLI | CLI-only |
| `--quality` | CLI | unchanged CLI | CLI-only |
| `--preset photo\|art` | CLI | unchanged CLI. The same ids as the picture presets | CLI-only |
| `--quality-tier fast\|max` | CLI | unchanged CLI. The same ids as the picture presets | CLI-only |
| `--model` | CLI | unchanged CLI | CLI-only |
| `--lossless` | CLI | unchanged CLI | CLI-only |
| `--double` | CLI | unchanged CLI | CLI-only |
| `--upscale` | CLI | unchanged CLI | CLI-only |
| `--json-progress` | CLI | unchanged CLI | CLI-only |
| `--dry-run` | CLI | unchanged CLI | CLI-only |
| `--lock-timeout` | CLI | unchanged CLI | CLI-only |
| `--parent-lock-pid` | CLI | unchanged CLI | CLI-only |
| `--capabilities` | CLI | unchanged CLI | CLI-only |
| `--json` | CLI | unchanged CLI | CLI-only |
| `--serve` / `--hidden` | CLI | unchanged CLI | CLI-only |
| `--register` / `--unregister` / `--exe` | CLI | unchanged CLI | CLI-only |
| `--install-gpu` | CLI | unchanged CLI | CLI-only |
| `--prepare-model` | CLI | unchanged CLI | CLI-only |
| `--install-suite` / `--uninstall-suite` | CLI | unchanged CLI | CLI-only |
| `--selftest` | CLI | unchanged CLI | CLI-only |
| `--version` | CLI | unchanged CLI | CLI-only |
| `--worker` | CLI | unchanged CLI | CLI-only |
| `upscale` subcommand | CLI | unchanged CLI | CLI-only |
