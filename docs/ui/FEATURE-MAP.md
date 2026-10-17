# Enlarge feature map

The main screen is `surface_for("enlarge")`. Simple chips, the Expert panel, the
shell's own More menu, and the batch wizard are the places a person can reach.
A row marked **CLI only, unchanged** stays on the command line. The program
still accepts the flag. It is not a menu item. A row marked **not run before
either, unchanged** was not run by the old window, and this window does not
show a control for it.

| Feature | Old location | New location | Status |
| --- | --- | --- | --- |
| Add images | Images step | wizard | mapped |
| Add folder | Images step | wizard | mapped |
| Remove | Images step | wizard | mapped |
| Clear | Images step | wizard | mapped |
| Include subfolders | Images step | wizard | mapped |
| Drop a full path | Window drag and drop | simple | mapped |
| Open one picture | Window header | drop area, click to choose | interim, until a shell command id |
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
| JPEG XL, lossless | Output step | simple | mapped |
| JPEG, HEIC | Output step | Expert | mapped |
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
| Compare before/after | Run step | simple | mapped |
| Before/after wipe | Run step | simple | mapped |
| Zoom | Run step | simple | mapped |
| Run | Run step | simple | mapped |
| Enlarge again | Run step | simple | mapped |
| Cancel | Run step | simple | mapped |
| Save | Run step | simple | mapped |
| Open folder | Run step | simple | mapped |
| Settings | Window header | Expert | mapped |
| Batch | Window header | More | interim, until a shell command id |
| Credits | Window header | More | interim, until a shell command id |
| About | — | More | interim, until a shell command id |
| Splash | Shared splash | simple | mapped |
| Face restore | Registered and not run | hidden | not run before either, unchanged |
| Generative upscale | Registered and not run | hidden | not run before either, unchanged |
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
| `--dry-run` | CLI | More | mapped |
| `--selftest` | CLI | More | mapped |
| `--version` | CLI | More | mapped |
| `--install-gpu` | CLI | More | mapped |
| `--prepare-model` | CLI | More | mapped |
| `--serve` | CLI | CLI | CLI only, unchanged |
| `--hidden` | CLI | CLI | CLI only, unchanged |
| `--register` | CLI | CLI | CLI only, unchanged |
| `--unregister` | CLI | CLI | CLI only, unchanged |
| `--exe` | CLI | CLI | CLI only, unchanged |
| `--install-suite` | CLI | CLI | CLI only, unchanged |
| `--uninstall-suite` | CLI | CLI | CLI only, unchanged |
| `--worker` | CLI | CLI | CLI only, unchanged |
| `--parent-lock-pid` | CLI | CLI | CLI only, unchanged |
| `--lock-timeout` | CLI | CLI | CLI only, unchanged |
| `--json` | CLI | CLI | CLI only, unchanged |
| `--json-progress` | CLI | CLI | CLI only, unchanged |
| `--capabilities` | CLI | CLI | CLI only, unchanged |

## What the shell does not offer

`surface_for("enlarge")` is an image surface. Its result actions are Save,
Open folder, Compare before/after, and Enlarge again. Its More menu is the
shell overflow: Open folder, Manage models, Settings, and Expert mode, plus
Enlarge a folder…, Credits, About, Preview output names, Check my GPU, Check
hash, and Remove weights.

Open, Batch, Credits, and About have no command id on this surface. Until
Components adds those ids, the drop area says “Drop a picture or click to
choose” and opens a file dialog, and Enlarge a folder…, Credits, and About are
extra actions on the existing More menu. Those four rows are marked interim.
There is no second toolbar.

Manage models is the shell's existing action. Install GPU and Prepare model go
through that window. They are not separate menu items.
