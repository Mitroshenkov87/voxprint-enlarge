# What Enlarge still needs from the shared shell

This window is `voxprint_common.ui.simple_image.SimpleLayout` and `SimpleWidget`.
It passes `surface_for("enlarge")` from `voxprint_common.ui.variants`. It does
not build its own `AppSurface`.

The pin is the one line in `requirements.txt`, from `suite/ui/ADOPTION.md`:

```text
voxprint-common[ui] @ git+https://github.com/Mitroshenkov87/voxprint-common@f4e95ccfa25780abaf50412cb095a7843725102a
```

Replace that line with the UI tag when Components publishes it.

`surface_for("enlarge")` is an image variant with a file drop and a picture
result. The action is Enlarge. Its built-in settings and preset file are the
picture-generation list plus a scale, and 4× stays unavailable until
`enlarge_installed`. This window keeps its own settings and
`enlarge/ui/presets/enlarge.json`, which ADOPTION allows. Picture chips write
registry ids from `enlarge/ui/presets/mdl-map.json` only.

## What the shell still does not draw

The shell overflow stays Open folder, Manage models, Settings, and Expert mode.
This window does not fork that menu. Its own simple actions and More menu sit
above the shell. `docs/ui/FEATURE-MAP.md` is the placement of each old control
and each CLI flag.

1. The enlarge surface's result is a picture. Compare, zoom, and the batch
   queue are extra screens. The shell picture and the history cells stay
   painted placeholders.
2. The image variant's action bar adds More like this and Use prompt again.
   Enlarge does not use a prompt. Add folder, Remove, and Clear are on the
   batch wizard. Open is the window's Open button. The shell still does not
   emit `file.open`.
3. The shell drop handler stores the file name. This window's filter runs
   first and keeps the full path. `open_ui` still uses `set_prompt_text`.
4. Lossless On cannot be limited to JPEG XL by the shell. The job refuses it
   for any other file type.
5. The shell preset file marks 4× unavailable. This app's size chips are both
   available. Four times as big is the default.
6. GPU choices are `auto`, `cuda:0`, and `cuda:1`.
7. Face restore and the generative tier are Expert choices. Running them needs
   a decision: the worker request has no field, and the registry says this
   build does not run them.
8. `--worker` has no window. Setup explains that and does not start it.
