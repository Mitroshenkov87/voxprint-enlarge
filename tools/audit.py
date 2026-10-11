"""Suite audit gate for VOXPrint AI Enlarge.

Runs ruff (F, E9, and D1 pydocstyle on the app and the engine), mypy, bandit (high severity), pip-audit, and gitleaks.
Vulture and radon are report-only. Exit code 1 when a blocking check fails.
A blocking tool that is not installed also fails the run.

    python tools/audit.py [--python PY] [--gitleaks-mode diff|full] [--gitleaks-range A..B]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "audit-report.txt"
ALLOWLIST = ROOT / "tools" / "audit-allowlist.toml"
PY_TARGETS = ["enlarge", "voxprint_engine", "main.py", "tools", "tests"]
BANDIT_TARGETS = ["enlarge", "voxprint_engine", "main.py", "tools"]


@dataclass
class Check:
    name: str
    summary: str
    detail: str
    blocks: bool


def _run(cmd: list[str], timeout: int = 600) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _tool(name: str) -> str | None:
    return shutil.which(name)


def _load_allowlist() -> list[tuple[str, str, str]]:
    raw = tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))
    rows = raw.get("advisory", [])
    if not isinstance(rows, list):
        raise ValueError("advisory must be a list")
    found: list[tuple[str, str, str]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"advisory {index} is not a table")
        ident = str(row.get("id") or "").strip()
        reason = str(row.get("reason") or "").strip()
        package = str(row.get("package") or "").strip().lower()
        if not ident or not reason:
            raise ValueError(f"advisory {index} needs an id and a reason")
        found.append((ident, reason, package))
    return found


def _allowed(ident: str, aliases: list[str], package: str, allow: list[tuple[str, str, str]]) -> bool:
    keys = {ident, *aliases}
    for entry_id, _reason, entry_pkg in allow:
        if entry_id not in keys:
            continue
        if entry_pkg and entry_pkg != package.lower():
            continue
        return True
    return False


def check_ruff() -> Check:
    if _tool("ruff") is None:
        return Check("ruff", "ruff is not installed", "", True)
    code, out, err = _run([sys.executable, "-m", "ruff", "check", *PY_TARGETS])
    text = (out + err).strip()
    return Check("ruff", "clean" if code == 0 else "findings", text, code != 0)


def check_mypy() -> Check:
    if _tool("mypy") is None:
        return Check("mypy", "mypy is not installed", "", True)
    code, out, err = _run([sys.executable, "-m", "mypy"])
    text = (out + err).strip()
    return Check("mypy", "clean" if code == 0 else "findings", text, code != 0)


def check_bandit() -> Check:
    if _tool("bandit") is None:
        return Check("bandit", "bandit is not installed", "", True)
    code, out, err = _run([sys.executable, "-m", "bandit", "-r", *BANDIT_TARGETS, "-f", "json", "-lll", "-q"])
    try:
        payload = json.loads(out or "{}")
    except ValueError:
        return Check("bandit", "could not parse bandit output", out + err, True)
    results = payload.get("results") or []
    high = [item for item in results if str(item.get("issue_severity", "")).lower() == "high"]
    if not high and code not in (0, 1):
        return Check("bandit", "bandit failed to run", err or out, True)
    lines = [f"{item.get('filename')}:{item.get('line')}: {item.get('issue_text')}" for item in high]
    return Check("bandit", "clean" if not high else f"{len(high)} high", "\n".join(lines), bool(high))


def _severity(vuln: dict[str, object]) -> str:
    raw = vuln.get("severity")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("score") is not None:
                try:
                    score = float(str(item["score"]))
                except ValueError:
                    continue
                if score >= 7.0:
                    return "high"
                if score >= 4.0:
                    return "medium"
                return "low"
    # A missing severity is treated as high so a parser miss cannot hide it.
    return "high"


def check_pip_audit(python: str) -> Check:
    if _tool("pip-audit") is None:
        return Check("pip-audit", "pip-audit is not installed", "", True)
    code, out, err = _run(
        [python, "-m", "pip_audit", "-r", "requirements.txt", "--format", "json", "--progress-spinner", "off"],
        timeout=900,
    )
    if code == 127 or "No module named" in err:
        return Check("pip-audit", "pip-audit is not installed", err, True)
    try:
        payload = json.loads(out or "{}")
    except ValueError:
        return Check("pip-audit", "could not parse pip-audit output", out + err, True)
    allow = _load_allowlist()
    deps = payload.get("dependencies", []) if isinstance(payload, dict) else []
    blocking: list[str] = []
    accepted: list[str] = []
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        name = str(dep.get("name") or "")
        for vuln in dep.get("vulns") or []:
            if not isinstance(vuln, dict):
                continue
            ident = str(vuln.get("id") or "")
            aliases = [str(item) for item in (vuln.get("aliases") or [])]
            level = _severity(vuln)
            line = f"{name} {ident} ({level})"
            if level not in ("high", "critical", "unknown"):
                continue
            if _allowed(ident, aliases, name, allow):
                accepted.append(line + " [allowlist]")
            else:
                blocking.append(line)
    summary = "clean" if not blocking else f"{len(blocking)} blocking"
    if accepted:
        summary += f", {len(accepted)} allowed"
    detail = "\n".join([*blocking, *accepted])
    if code not in (0, 1) and not blocking:
        return Check("pip-audit", "pip-audit failed to run", err or out, True)
    return Check("pip-audit", summary, detail, bool(blocking))


def check_gitleaks(mode: str, commit_range: str) -> Check:
    binary = _tool("gitleaks")
    if binary is None:
        return Check("gitleaks", "gitleaks is not installed", "", True)
    cmd = [binary, "detect", "--source", str(ROOT), "--redact", "--no-banner", "--exit-code", "1"]
    if mode == "diff" and commit_range:
        cmd.extend(["--log-opts", commit_range])
    code, out, err = _run(cmd, timeout=600)
    text = (out + err).strip()
    if code == 0:
        return Check("gitleaks", f"clean ({mode})", text, False)
    return Check("gitleaks", f"findings ({mode})", text, True)


def check_report_only(name: str, args: list[str]) -> Check:
    if _tool(name) is None:
        return Check(name, "not installed (report only)", "", False)
    _code, out, err = _run([sys.executable, "-m", name, *args])
    return Check(name, "report", (out + err).strip()[:4000], False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable, help="interpreter whose environment pip-audit checks")
    parser.add_argument("--gitleaks-mode", choices=("full", "diff"), default="full")
    parser.add_argument("--gitleaks-range", default="")
    args = parser.parse_args(argv)
    checks = [
        check_ruff(),
        check_mypy(),
        check_bandit(),
        check_pip_audit(args.python),
        check_gitleaks(args.gitleaks_mode, args.gitleaks_range),
        check_report_only("vulture", ["enlarge", "voxprint_engine", "main.py", "tools"]),
        check_report_only("radon", ["cc", "-s", "enlarge", "voxprint_engine"]),
    ]
    lines = ["VOXPrint AI Enlarge audit", ""]
    failed = False
    for check in checks:
        mark = "FAIL" if check.blocks else "ok"
        if check.blocks:
            failed = True
        lines.append(f"[{mark}] {check.name}: {check.summary}")
        if check.detail:
            lines.append(check.detail.rstrip())
            lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(text, end="")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text("```\n" + text + "```\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
