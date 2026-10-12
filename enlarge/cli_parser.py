"""The ``voxprint-enlarge`` argument parser.

``tools/generate_cli_reference.py`` renders this parser into the docs site.
``docs/CLI.md`` is the contract prose. Defaults live here so the page and the
process stay the same.
"""
from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from enlarge.ipc.protocol import dumps, error_message


class CliParser(argparse.ArgumentParser):
    """Parser that exits with code 2 instead of raising into the caller.

    When ``json_errors`` is set, an argument error is one ``bad_input`` JSON
    line on stdout. Without that flag the short message stays on stderr.
    """

    json_errors: bool = False

    def error(self, message: str) -> NoReturn:
        """Exit 2 with the short argparse message."""
        self.exit(2, f"{self.prog}: error: {message}\n")

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Status 2 with ``json_errors`` writes one JSON object and no usage text."""
        if status == 2 and self.json_errors:
            text = (message or "invalid arguments").strip()
            prefix = f"{self.prog}: error: "
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            payload = dumps(error_message("", "bad_input", text))
            buffer = getattr(sys.stdout, "buffer", None)
            if buffer is not None:
                buffer.write(payload)
                buffer.flush()
            else:
                sys.stdout.write(payload.decode("utf-8"))
                sys.stdout.flush()
            raise SystemExit(2)
        if message:
            self._print_message(message, sys.stderr)
        raise SystemExit(status)


def _add_job_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=None, metavar="FILE", help="Source image.")
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        default="",
        metavar="FILE",
        help="Destination file. --out is the same flag.",
    )
    parser.add_argument("--output-dir", default="", metavar="DIR", help="Directory for the default output name.")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument(
        "--scale",
        type=int,
        choices=(2, 4),
        default=None,
        help="Model scale, 2 or 4. When scale and long-side are both omitted, the scale is 4.",
    )
    scale.add_argument("--long-side", type=int, default=None, metavar="PX", help="Target long side in pixels.")
    parser.add_argument(
        "--format",
        default="",
        metavar="FMT",
        help="avif (default), jxl, webp, jpeg, or heic when a system encoder is present.",
    )
    parser.add_argument("--lossless", action="store_true", help="Lossless JPEG XL.")
    parser.add_argument(
        "--quality",
        default=None,
        help="Encode quality 1-100 (default 60). fast or max selects the model tier when --quality-tier is omitted.",
    )
    parser.add_argument(
        "--preset",
        choices=("photo", "art"),
        default="photo",
        help="photo (default) or art. An explicit --model overrides this.",
    )
    parser.add_argument("--model", default="", metavar="ID", help="Registry model id. Overrides preset and quality tier.")
    parser.add_argument(
        "--quality-tier",
        choices=("fast", "max"),
        default="fast",
        help="Model tier. Default fast.",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help="Seconds to wait for the suite GPU lock. Default 600.",
    )
    parser.add_argument(
        "--parent-lock-pid",
        type=int,
        default=None,
        metavar="PID",
        help="Pid that already holds the suite GPU lock. VOXPRINT_GPU_PARENT_PID is the same value.",
    )
    parser.add_argument("--json-progress", action="store_true", help="Required for a headless job. Stdout is JSON lines.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan. No model, no GPU, no output file.",
    )


def build_parser(json_errors: bool = False) -> CliParser:
    """Return the process-wide parser. The docs build calls this and prints its help."""
    parser = CliParser(
        prog="voxprint-enlarge",
        description="Enlarge a still image. Headless jobs print JSON lines and do not open a window.",
    )
    parser.json_errors = json_errors
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    parser.add_argument("--worker", action="store_true", help="Internal inference process.")
    parser.add_argument("--capabilities", action="store_true", help="Print what this install can run.")
    parser.add_argument("--json", action="store_true", help="JSON for --capabilities and --selftest.")
    parser.add_argument("--serve", action="store_true", help="Listen for protocol 1.")
    parser.add_argument("--hidden", action="store_true", help="With --serve, leave the window closed.")
    parser.add_argument("--register", action="store_true", help="Write discovery records for this user.")
    parser.add_argument("--unregister", action="store_true", help="Remove those discovery records.")
    parser.add_argument("--exe", default=None, metavar="PATH", help="Executable path stored by --register.")
    parser.add_argument(
        "--install-gpu",
        action="store_true",
        help="Install torch, the CUDA 13 ONNX Runtime, and TensorRT.",
    )
    parser.add_argument(
        "--prepare-model",
        default="",
        metavar="ID",
        help="Convert one shared model to ONNX. The MDL post_install hook runs this.",
    )
    parser.add_argument(
        "--install-suite",
        action="store_true",
        help="Install shared Python 3.14, the CUDA wheels, and the default model.",
    )
    parser.add_argument(
        "--uninstall-suite",
        action="store_true",
        help="Drop this app's hold on shared files. Delete a file only when nothing else uses it.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Re-check the shared install and repair it. With --json, print one object.",
    )
    parser.add_argument("--double", action="store_true", help="Legacy --upscale only. Run a second pass.")
    parser.add_argument(
        "--upscale",
        default=None,
        metavar="FILE",
        help="Legacy command. Prints a path for a person and needs --out.",
    )
    _add_job_flags(parser)
    commands = parser.add_subparsers(dest="command", parser_class=CliParser)
    upscale = commands.add_parser("upscale", help="Headless job. Same flags as voxprint-enlarge itself.")
    if isinstance(upscale, CliParser):
        upscale.json_errors = json_errors
    _add_job_flags(upscale)
    return parser


def parse_cli(argv: list[str]) -> argparse.Namespace | int:
    """Parse ``argv``. An int is the exit code argparse already reported."""
    parser = build_parser(json_errors="--json-progress" in argv)
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        try:
            return int(code)
        except (TypeError, ValueError):
            return 2
