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

## Missing on the shell

1. The enlarge surface's result is a picture. The migration plan's compare
   slot, the divider, and a batch list are not on `surface_for("enlarge")`.
   The picture and the history cells are painted placeholders. There is no
   API for the opened file and the enlarged file.
2. The image variant's action bar adds More like this and Use prompt again.
   Enlarge does not use a prompt. Add images, Add folder, Remove, Clear,
   Browse, and Beside source are not controls on this surface. Open is
   `file.open`, and nothing here emits it.
3. The drop handler stores the file name, not the path
   (`SimpleWidget._install_drop`). `open_ui` still puts a full path in the
   prompt through `set_prompt_text`.
4. Manage models is the overflow item `models.manage`. Credits, check hash,
   and remove weights have no command on that menu.
5. Lossless On cannot be enabled only for JPEG XL. The job refuses it for any
   other file type. Include-subfolders is stored and is not applied, because
   there is no folder list.
6. The shell preset file marks 4× unavailable. This app's size chips are both
   available. Four times as big is the default.
7. GPU choices are `auto`, `cuda:0`, and `cuda:1`.
