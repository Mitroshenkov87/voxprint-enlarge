"""Write the CLI reference from the argparse parser, and copy the roadmap into the site tree.

The docs workflow runs this before ``mkdocs build``. ``docs/CLI.md`` stays the contract prose.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from enlarge.cli_parser import build_parser


def _default_rows(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for action in parser._actions:
        if action.dest in {"help", "command"}:
            continue
        if action.default in (None, argparse.SUPPRESS, False, ""):
            continue
        option = action.option_strings[0] if action.option_strings else action.dest
        rows.append((option, str(action.default)))
    return rows


def render() -> str:
    """Markdown for the reference page. The help text is ``format_help()`` of the live parser."""
    parser = build_parser()
    help_text = parser.format_help().rstrip()
    lines = ["| Flag | Default |", "|---|---|"]
    for option, default in _default_rows(parser):
        lines.append(f"| `{option}` | `{default}` |")
    table = "\n".join(lines)
    return f"""# CLI reference

Generated from `enlarge.cli_parser.build_parser` when the docs site is built.
The contract prose is [the CLI contract](../CLI.md).

When `--scale` and `--long-side` are both omitted, the scale is 4. That default is applied after parsing, so it is not a flag default in the table.

```text
{help_text}
```

## Defaults

{table}
"""


def generate(dest: Path | None = None) -> Path:
    """Write ``cli-reference.md`` and a copy of the root roadmap. Return the reference path."""
    folder = dest if dest is not None else ROOT / "docs" / "generated"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "cli-reference.md"
    target.write_text(render(), encoding="utf-8")
    shutil.copyfile(ROOT / "ROADMAP.md", folder / "roadmap.md")
    return target


def main() -> None:
    """Write the pages under ``docs/generated``."""
    generate()


if __name__ == "__main__":
    main()
