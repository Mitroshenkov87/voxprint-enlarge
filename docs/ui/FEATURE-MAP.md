# Enlarge feature map

The main screen is `surface_for("enlarge")`. Everything else is a simple action,
the Expert panel, the More menu, or the batch wizard. Nothing in the old window
or the CLI is dropped. A row marked **needs decision** is visible, and the
behaviour still needs a decision before it can run.

| Feature | Old location | New location | Status |
| --- | --- | --- | --- |
| Add images | Images step | simple | mapped |
| Add folder | Images step | wizard | mapped |
| Remove | Images step | wizard | mapped |
| Clear | Images step | wizard | mapped |
| Include subfolders | Images step | simple | mapped |
| Drop a full path | Window drag and drop | simple | mapped |
| Photographs | Model step | simple | mapped |
| Photographs, highest detail | Model step | simple | mapped |
| Drawings | Model step | simple | mapped |
| Manage models | Model step | More | mapped |
| Check hash | Model step | More | mapped |
| Remove weights | Model step | More | mapped |
| Model id | Model step | Expert | mapped |
| Licence | Model step | Expert | mapped |
| Scale 2 and 4 | Output step | simple | mapped |
| Double upscale | Output step | Expert | mapped |
| Long side | CLI `--long-side` | Expert | mapped |
| AVIF, JPEG XL, WebP | Output step | simple | mapped |
| JPEG, HEIC | Output step | Expert | mapped |
| Lossless JPEG XL | Output step | simple | mapped |
| Quality | Output step | Expert | mapped |
| Output folder | Output step | Expert | mapped |
| Beside the source | Output step | Expert | mapped |
| Tile size | Output step and Settings | Expert | mapped |
| TensorRT note | Output step | wizard | mapped |
| GPU | Settings | Expert | mapped |
| GPU warning | Output step | simple | mapped |
| Custom models folder | Settings | Expert | mapped |
| Shared models folder | Settings | Expert | mapped |
| Queue | Run step | wizard | mapped |
| Queue restore | Run step | wizard | mapped |
| Compare | Run step | simple | mapped |
| Zoom | Run step | simple | mapped |
| Run | Run step | simple | mapped |
| Cancel | Run step | simple | mapped |
| Settings | Window header | Expert | mapped |
| Credits | Window header | simple | mapped |
| Splash | Shared splash | simple | mapped |
| Face restore choice | Registry role face | Expert | mapped |
| Run face restore | Registered and not run | Expert | needs decision |
| Generative choice | Registry role generative | Expert | mapped |
| Run generative upscale | Registered and not run | Expert | needs decision |
| `--input` | CLI | simple | mapped |
| `--output` / `--out` | CLI | Expert | mapped |
| `--output-dir` | CLI | Expert | mapped |
| `--scale` | CLI | simple | mapped |
| `--long-side` | CLI | Expert | mapped |
| `--format` | CLI | simple | mapped |
| `--lossless` | CLI | simple | mapped |
| `--quality` | CLI | Expert | mapped |
| `--preset` | CLI | simple | mapped |
| `--quality-tier` | CLI | Expert | mapped |
| `--model` | CLI | Expert | mapped |
| `--double` | CLI | Expert | mapped |
| `--upscale` | CLI | simple | mapped |
| `upscale` subcommand | CLI | simple | mapped |
| `--json-progress` | CLI | Expert | mapped |
| `--dry-run` | CLI | More | mapped |
| `--lock-timeout` | CLI | Expert | mapped |
| `--parent-lock-pid` | CLI | Expert | mapped |
| `--capabilities` | CLI | More | mapped |
| `--json` | CLI | More | mapped |
| `--serve` | CLI | More | mapped |
| `--hidden` | CLI | More | mapped |
| `--register` | CLI | More | mapped |
| `--unregister` | CLI | More | mapped |
| `--exe` | CLI | More | mapped |
| `--install-gpu` | CLI | More | mapped |
| `--prepare-model` | CLI | More | mapped |
| `--install-suite` | CLI | More | mapped |
| `--uninstall-suite` | CLI | More | mapped |
| `--selftest` | CLI | More | mapped |
| `--version` | CLI | More | mapped |
| `--worker` | CLI | More | needs decision |

## Needs a decision

* **Run face restore.** Expert stores the registry id. The worker request has no field for it, and this build's registry says the step is not run. The job says so and still enlarges the picture.
* **Run generative upscale.** The same limit, for the generative registry row.
* **`--worker`.** More → Setup shows why. The worker is the internal inference process. Starting it in this window would replace the screen.
