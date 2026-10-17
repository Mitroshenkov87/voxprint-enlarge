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

## Slots this surface does not have

The shell overflow is Open folder, Manage models, Settings, and Expert mode.
There is one More button, that one. Enlarge adds Enlarge a folder…, Credits,
About, Preview file names, and Check my GPU to that menu. Expert mode stays
because the shell adds it. Check hash and Remove weights are on the Manage
models screen, and removing weights asks for confirmation. There is no second
toolbar.

The image result bar is hardcoded as More like this, Save, Open folder, and
Use prompt again. Enlarge retargets the first and the last to Compare
before/after and Enlarge again. Save and Open folder stay.

These four are interim until Components adds command ids. The rows in
`docs/ui/FEATURE-MAP.md` say so.

1. **Open.** `file.open` is emitted by the viewer bar only. This surface does
   not emit it. The drop area says “Drop a picture or click to choose” and
   opens a file dialog. A drop still puts the full path in the file area.
2. **Batch.** No command id and no result-bar slot. More → Enlarge a folder…
   opens the wizard, and so does dropping a folder or more than one file.
3. **Credits.** No command id and no result-bar slot. More → Credits opens the
   screen.
4. **About.** No command id. More → About shows the version until the shell
   has an About slot.
5. **A real wipe.** The enlarge result kind is `PICTURE`. `COMPARE` paints a
   decorative split from `_wipe` and does not take the opened file or the
   enlarged file. This window draws that split on the result label after a
   run, with the shell's `compareSlider` line. A shell API that accepts the
   two pixmaps would replace that.
6. **Chip and spin metrics.** `_fit_chip` measures before the shell font is
   applied, so "Twice as big", "Four times as big", and «Режим эксперта» clip.
   This window measures again after polish, with the same DemiBold advance
   plus 14 px of padding on each side. Expert spins are created at 72 px, which
   cuts off the number. This window reserves 120 px. A long side of 0 shows
   the special value Off. The shell stylesheet does not paint scrollbar arrows,
   so the native arrows look broken. This window draws a handle and leaves the
   arrows out.
7. **Manage models** does not run `--install-gpu` or `--prepare-model` itself.
   Those flags stay on the command line and are reached through that existing
   More action, as asked. A model-manager action that installs the GPU runtime
   and prepares one model id would make the path real.

## What stays as it is

The shell drop handler stores the file name. This window's filter runs first
and keeps the full path. `open_ui` still uses `set_prompt_text`.

The shell preset file marks 4× unavailable. This app's size chips are both
available. Four times as big is the default.

GPU choices are `auto`, `cuda:0`, and `cuda:1`.

Face restore and the generative tier are hidden. The old window did not run
them, and the worker request has no field for them.

`--serve`, `--hidden`, `--register`, `--unregister`, `--exe`,
`--install-suite`, `--uninstall-suite`, `--worker`, `--parent-lock-pid`,
`--lock-timeout`, `--json`, `--json-progress`, and `--capabilities` stay on
the command line.
