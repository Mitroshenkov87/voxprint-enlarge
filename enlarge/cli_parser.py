"""The ``voxprint-enlarge`` argument parser.

``tools/generate_cli_reference.py`` renders this parser into the docs site.
``docs/CLI.md`` is the contract prose. Defaults live here so the page and the
process stay the same.
"""
from __future__ import annotations

import argparse
from typing import NoReturn


class CliParser(argparse.ArgumentParser):
    """Parser that exits with code 2 instead of raising into the caller."""

    def error(self, message: str) -> NoReturn:
        """Print the argparse message and exit 2."""
        self.exit(2, f"{self.prog}: error: {message}\n")


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


def build_parser() -> CliParser:
    """Return the process-wide parser. The docs build calls this and prints its help."""
    parser = CliParser(
        prog="voxprint-enlarge",
        description="Enlarge a still image. Headless jobs print JSON lines and do not open a window.",
    )
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    parser.add_argument("--worker", action="store_true", help="Internal inference process.")
    parser.add_argument("--capabilities", action="store_true", help="Print what this install can run.")
    parser.add_argument("--json", action="store_true", help="Required with --capabilities.")
    parser.add_argument("--serve", action="store_true", help="Listen for protocol 1.")
    parser.add_argument("--hidden", action="store_true", help="With --serve, leave the window closed.")
    parser.add_argument("--register", action="store_true", help="Write discovery records for this user.")
    parser.add_argument("--unregister", action="store_true", help="Remove those discovery records.")
    parser.add_argument("--exe", default=None, metavar="PATH", help="Executable path stored by --register.")
    parser.add_argument(
        "--install-gpu",
        action="store_true",
        help="Install the CUDA 13 ONNX Runtime and TensorRT wheels.",
    )
    parser.add_argument("--selftest", action="store_true", help="Open the window, then quit.")
    parser.add_argument("--double", action="store_true", help="Legacy --upscale only. Run a second pass.")
    parser.add_argument(
        "--upscale",
        default=None,
        metavar="FILE",
        help="Legacy command. Prints a path for a person and needs --out.",
    )
    _add_job_flags(parser)
    commands = parser.add_subparsers(dest="command")
    upscale = commands.add_parser("upscale", help="Headless job. Same flags as voxprint-enlarge itself.")
    _add_job_flags(upscale)
    return parser


def parse_cli(argv: list[str]) -> argparse.Namespace | int:
    """Parse ``argv``. An int is the exit code argparse already reported on stderr."""
    parser = build_parser()
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
