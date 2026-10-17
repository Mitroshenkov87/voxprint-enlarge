# What Enlarge still needs from the shared shell

This window is `voxprint_common.ui.simple_image.SimpleLayout` and `SimpleWidget`.
It does not copy that screen. The pin is the `voxprint-common` line in
`requirements.txt` (commit `f4e95ccfa25780abaf50412cb095a7843725102a`,
voxprint-common PR #13, branch `cursor/expert-mode-3cae`).

`surface_for("enlarge")` is not what this window passes. That surface is an
image file in and a still picture out. Its settings are the picture-generation
list plus `image.scale`. Its preset file is only 2× and 4×, and 4× stays
unavailable until `enlarge_installed`, with the action `install.enlarge`.
Enlarge is the app that enlarges, so that gate is backwards here.

## Configuration this window passes

`ENLARGE_SURFACE` in `enlarge/ui/simple_window.py`:

- input: `InputKind.FILE`, drop text `simple.drop.picture`
- result: `ResultKind.COMPARE`, the same wipe Clarity uses
- action label: `simple.action.enlarge` ("Enlarge")
- variant: `VariantId.VIDEO`

The variant is `video` only because the widget picks the action bar from the
variant. `VariantId.IMAGE` adds "More like this" and "Use prompt again".
`VariantId.VIDEO` leaves Save and Open folder. The input is still a picture,
and the expert fields are Enlarge's, not the video encoder list.

Presets and expert values are one `SettingState`. The preset file is
`enlarge/ui/presets/enlarge.json`. Chips write registry ids:

| Chip | Registry id |
| --- | --- |
| Photo | `4x-nomos2-realplksr-dysample` |
| Drawing or anime | `realesrgan-x4plus-anime6b` |
| Old or blurry photo | `4x-nomos2-hq-dat2` |

Bigger file writes `enlarge.scale` `2` or `4`. A hand edit of a key that chip
owns shows "{name} + changes" and Reset, from the shell. Expert mode is
`ui.expert_mode` in suite settings. Turning it on or off does not write the
picture values.

Model choices are the ids from `mdl.json` via `voxprint_common.mdl.catalog.load_manifest`.
They are not weight file names. A new id needs a catalog line
`enlarge.model.<id>` in `enlarge/ui/locales/en.json` and `ru.json`, and a
preset row only if a chip should select it.

## Missing on the shell

1. An enlarge surface: file in, compare wipe out, action bar of Open, Enlarge,
   Save, and Manage models. No "More like this", no "Use prompt again", and no
   video encoder or CRF fields. `surface_for("enlarge")`, `settings_for`, and
   `variant_presets` should grow that surface so Enlarge can stop passing its
   own `AppSurface`.
2. An Open button on a file input. Open exists on the viewer bar
   (`file.open`). A file drop has no button. This window handles `file.open`,
   and nothing on this surface emits it. The visible control is the drop
   target.
3. The drop handler stores the file name, not the path
   (`SimpleWidget._install_drop`). A dropped file cannot be opened again.
   `set_prompt_text` can hold a full path, which is what another app's
   `open_ui` uses.
4. The compare wipe paints a decorative before and after (`_wipe` / `_scene`).
   There is no way to give it the opened picture and the enlarged file.
   History cells for a compare result are numbers, not the picture.
5. Manage models is the overflow item `models.manage`, not a button on the
   main row. This window runs `voxprint-mdl --gui --app enlarge` for that
   command.
6. Face restore and denoise are in the shared settings state. The worker
   request has no field for either, and this branch does not change the
   engine. The expert help text says the choice is stored and not applied.
7. The shell's enlarge preset marks 4× unavailable. This app's size chips are
   both available. x2 is the default.
8. GPU is the suite value (`auto`, `cuda:0`, `cuda:1`). A stored `cuda:N`
   past the second card is not a choice on this panel.
