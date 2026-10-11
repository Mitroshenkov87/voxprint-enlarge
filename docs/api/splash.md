# Splash

Shared 960 by 540 splash. Import this module only from a windowed app.
`voxprint_engine` and `voxprint_engine.api` do not import it.

```python
from pathlib import Path

from voxprint_engine.ui.splash import SplashScreen

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

::: voxprint_engine.ui.splash.SplashScreen
    options:
      show_root_heading: true
      members:
        - show
        - set_stage
        - set_progress
        - finish
        - __enter__
        - __exit__
      docstring_style: google
      show_bases: false
