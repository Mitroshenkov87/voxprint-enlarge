# What Enlarge still needs from the shared shell

This window is `voxprint_common.ui.simple_image.SimpleLayout` and `SimpleWidget`.
It does not copy that screen.

The migration plan (`suite/ui/MIGRATION-PLAN.md` on project-notes
`cursor/ui-migration-plan-0aba`, PR #2) says apps pin the shell tag and do not
pin the PR branch. That tag is not published. The one line in `requirements.txt`
is voxprint-common PR #13 head `0d1e30de68cc49549a2a94cb3695e5601c17e7db`, which
is the shell `docs/ui.md` describes. Replace that line with the tagged wheel.

`surface_for("enlarge")` is still an image file in and a still picture out.
Its settings are the picture-generation list plus `image.scale`. Its preset
file is only 2× and 4×, and 4× stays unavailable until `enlarge_installed`.
The plan's "Before Enlarge" row asks for a file input, a batch result, a
compare slot, and preset rows that write model id, scale, and format, with no
required prompt. That surface is not in this commit.

## Configuration this window passes

`ENLARGE_SURFACE` in `enlarge/ui/simple_window.py`:

- input: `InputKind.FILE`, drop text `simple.drop.picture`
- result: `ResultKind.COMPARE`, the same wipe Clarity uses
- action label: `simple.action.enlarge` ("Enlarge")
- variant: `VariantId.VIDEO`

The variant is `video` only so the action bar stays Save and Open folder.
`VariantId.IMAGE` adds "More like this" and "Use prompt again".

Presets and expert values are one `SettingState`. Picture chips write registry
ids from `enlarge/ui/presets/mdl-map.json`. That file is the only copy of those
ids. They are placeholders until Model Forge replaces them. IPC `photo`/`fast`,
`photo`/`max`, and `art` read the same file.

| Chip | IPC | Placeholder registry id |
| --- | --- | --- |
| Photographs | `photo` / `fast` | `4x-nomos2-realplksr-dysample` |
| Photographs, highest detail | `photo` / `max` | `4x-nomos2-hq-dat2` |
| Drawings | `art` / `fast` and `art` / `max` | `realesrgan-x4plus-anime6b` |

Size writes 2 or 4. Four times as big is the default. File type writes AVIF,
JPEG XL, or WebP. AVIF is the default. Lossless writes a bool. Subfolders
writes a bool and defaults to on. Every picture chip also writes quality 60.

## Missing on the shell

1. An enlarge surface: files in, compare wipe out, no required prompt, and an
   action bar of Open, Add images, Add folder, Remove, Clear, Enlarge, Save,
   and Manage models. No "More like this", no "Use prompt again", and no video
   encoder or CRF. `surface_for("enlarge")` should grow that surface so Enlarge
   can stop passing its own `AppSurface`.
2. Simple actions the plan lists and this screen cannot show: Add images, Add
   folder, Remove, Clear, Browse for an output folder, and Beside source.
   Beside source and the output folder are expert values until those buttons
   exist. Open exists as `file.open`, and nothing on this surface emits it.
3. The drop handler stores the file name, not the path
   (`SimpleWidget._install_drop`). A dropped file cannot be opened again.
   `set_prompt_text` can hold a full path, which is what another app's
   `open_ui` uses.
4. The compare wipe paints a decorative before and after (`_wipe` / `_scene`).
   There is no way to give it the opened picture and the enlarged file.
   History cells for a compare result are numbers, not the picture. A batch
   result list is not on this surface.
5. Manage models is the overflow item `models.manage`, not a button on the
   main row. This window runs `voxprint-mdl --gui --app enlarge` for that
   command. Credits, check hash, and remove weights have no command on the
   overflow menu.
6. Lossless On cannot be enabled only when the file type is JPEG XL. Both
   chips stay available. The job refuses lossless for any other file type.
   Include-subfolders is stored and is not applied, because there is no folder
   list yet.
7. The shell's enlarge preset marks 4× unavailable. This app's size chips are
   both available. Four times as big is the default.
8. GPU choices are `auto`, `cuda:0`, and `cuda:1`. JPEG and HEIC stay on the
   expert file-type list. The expert model control shows registry ids.
