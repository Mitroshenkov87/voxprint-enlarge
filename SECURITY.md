# Security

Voxprint AI Enlarge runs on your computer. It does not upload the pictures you enlarge.

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/Mitroshenkov87/voxprint-enlarge/security/advisories/new) for a security issue. Do not open a public issue for a vulnerability that is not yet fixed.

Include the version from `voxprint-enlarge --version`, the operating system, and a short description of the impact. If you can, add the smallest change that shows the problem.

## Supported versions

Security fixes land on `main` and ship in the next release. The current pre-release is the one listed on the [releases page](https://github.com/Mitroshenkov87/voxprint-enlarge/releases).

## Dependencies

Runtime dependencies are pinned in `requirements.txt`. A GPU install uses the CUDA wheels named in `docs/ENGINE.md`. Please report a vulnerable dependency the same way as any other vulnerability.
