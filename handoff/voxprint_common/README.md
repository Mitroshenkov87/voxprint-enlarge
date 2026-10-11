# voxprint_common

Suite manifest and splash. This folder is not part of `voxprint_engine` and the engine does not import it.

Enlarge runs the same files from `enlarge/_vendor/voxprint_common/`.

## manifest.py

`SharedManifest`, `SharedManifestError`, `load`, `save`, `update`, `find_reusable`, `reserve`, `add_ref`, `remove_ref`, `lock_hash`.

`SCHEMA` is 1. `PYTHON_LOCK`, `TORCH_LOCK`, `ORT_LOCK`, and `TRT_LOCK` are the pin strings whose sha256 is stored on a component.

`load` reads `manifest.json`, migrates an old Canvas or Enlarge file, and writes schema 1 back. `update` holds `manifest.lock` and saves when the body returns. `save` writes `manifest.json.tmp`, fsyncs, and renames it. `find_reusable` matches id, version, and sha256. `reserve` records a side-by-side path instead of overwriting a component another app refs. `remove_ref` returns true when `refs` is empty.

Importing this module does not import Qt.

## splash.py

`SplashScreen(app_id, art_path, identity_color, build_json_path=None)`, `show()`, `set_stage(text)`, `set_progress(fraction or None)`, `finish(main_window)`, and a context manager.

The window is 960 by 540, borderless, and centred, with 12 px corners, a drop shadow, and artwork scaled to cover. A sibling `<stem>@2x<suffix>` is used when the screen device pixel ratio is at least 2 and that file exists. The left third is inset 48 px: the name at 40 px, the version line at 16 px and 70% off-white, the stage line at 14 px, and a 4 px bar. The version line is `X.Y.Z RC N build B "Codename"` from `BUILD.json`, without `RC N` for a final version. The bar shimmers until `set_progress` is given a fraction. `finish` waits until 600 ms have passed since `show`, then fades out over 200 ms.

```python
from pathlib import Path

from handoff.voxprint_common.splash import SplashScreen

splash = SplashScreen(
    "Voxprint AI Enlarge",
    Path("assets/splash.jpg"),
    "#ff6b8a",
    Path("BUILD.json"),
)
with splash:
    splash.set_stage("Loading UI")
    splash.set_progress(0.2)
    splash.finish(window)
```

Importing this module imports PySide6.
