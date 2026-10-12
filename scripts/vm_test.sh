#!/usr/bin/env bash
# Non-interactive GPU checks for a VOXPrint AI Enlarge machine.
# Exit 0 pass, 1 a test failed, 2 setup, 3 no usable GPU.
# --smoke is the short set (selftest, one stock upscale and its cached rerun,
# anime 6B at 4x, and the long-side chain). It is also the default.
#
# The interpreter is interpreter.json (or the manifest python path), then a
# python that is really in that folder, then `py -3`. `python3` is never
# used: on Windows it is often the Microsoft Store alias. The chosen command
# is printed as `python: ...`.
set -u

# Git bash does not open C:\Windows paths. Forward slashes do.
_to_unix() {
  local value="$1"
  printf '%s' "${value//\\//}"
}

# Print one interpreter command. A path is one line. `py -3` is the launcher.
# Exit 1 when no shared interpreter and no working `py -3` exist.
resolve_vm_python() {
  local root
  while IFS= read -r root; do
    [[ -n "$root" ]] || continue
    if _resolve_shared_tree "$root"; then
      return 0
    fi
  done < <(_shared_roots)
  # A dead `py -3` exits 1. The subshell keeps `set -e` from stopping the script.
  if command -v py >/dev/null 2>&1; then
    if ( py -3 -c "import sys" ) >/dev/null 2>&1; then
      printf '%s\n' "py -3"
      return 0
    fi
  fi
  return 1
}

_shared_roots() {
  local data home
  if [[ -n "${VOXPRINT_HOME:-}" ]]; then
    printf '%s\n' "$(_to_unix "$VOXPRINT_HOME")/shared"
  fi
  if [[ -n "${LOCALAPPDATA:-}" ]]; then
    printf '%s\n' "$(_to_unix "$LOCALAPPDATA")/Voxprint/shared"
  fi
  data="$(_to_unix "${XDG_DATA_HOME:-${HOME:-}/.local/share}")"
  home="$data/voxprint/shared"
  if [[ -d "$home" ]]; then
    printf '%s\n' "$home"
  fi
}

_resolve_shared_tree() {
  local root="$1" rel dir name
  if [[ -f "$root/manifest.json" ]]; then
    while IFS= read -r rel; do
      [[ -n "$rel" ]] || continue
      if _consider_pointer "$root/$rel"; then
        return 0
      fi
    done < <(sed -n 's/.*"path"[[:space:]]*:[[:space:]]*"\(python[^"]*\)".*/\1/p' "$root/manifest.json")
  fi
  shopt -s nullglob
  for dir in "$root"/python "$root"/python-*; do
    [[ -d "$dir" ]] || continue
    name="$(basename "$dir")"
    case "$name" in
      python|python-*)
        if _consider_pointer "$dir"; then
          shopt -u nullglob
          return 0
        fi
        ;;
    esac
  done
  shopt -u nullglob
  return 1
}

_consider_pointer() {
  local dir="$1" exe version
  if [[ -f "$dir/interpreter.json" ]]; then
    exe="$(_json_field "$dir/interpreter.json" executable)"
    version="$(_json_field "$dir/interpreter.json" version)"
    exe="$(_unescape_json "$exe")"
    exe="$(_to_unix "$exe")"
    case "$version" in
      3.14|3.14.*)
        if [[ -n "$exe" && -f "$exe" ]]; then
          printf '%s\n' "$exe"
          return 0
        fi
        ;;
    esac
  fi
  for exe in "$dir/python.exe" "$dir/pythonw.exe" "$dir/bin/python3" "$dir/Scripts/python.exe"; do
    if [[ -f "$exe" ]]; then
      printf '%s\n' "$exe"
      return 0
    fi
  done
  return 1
}

_json_field() {
  local file="$1" field="$2"
  sed -n "s/.*\"${field}\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" "$file" | head -n 1
}

_unescape_json() {
  local value="$1"
  value="${value//\\\\/\\}"
  printf '%s' "$value"
}

vm_test_main() {
  local SMOKE=1
  if [[ "${1:-}" == "--smoke" ]]; then
    shift
  elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "usage: scripts/vm_test.sh [--smoke]" >&2
    exit 2
  elif [[ $# -gt 0 ]]; then
    echo "setup: unknown argument: $1" >&2
    exit 2
  fi

  if [[ "$SMOKE" -ne 1 ]]; then
    echo "setup: nothing to run" >&2
    exit 2
  fi

  local choice
  if ! choice="$(resolve_vm_python)"; then
    echo "setup: shared runtime interpreter was not found" >&2
    exit 2
  fi
  echo "python: $choice" >&2
  PYTHON=()
  if [[ "$choice" == "py -3" ]]; then
    PYTHON=(py -3)
  else
    PYTHON=("$choice")
  fi

  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  OUT="${VOXPRINT_VM_OUT:-$(mktemp -d)}"
  mkdir -p "$OUT" || { echo "setup: cannot create $OUT" >&2; exit 2; }

  APP=()
  local win_root="" win_cli="" legacy="" env_value
  for env_value in "${ProgramW6432:-}" "${PROGRAMW6432:-}" "${ProgramFiles:-}" "${PROGRAMFILES:-}"; do
    if [[ -n "$env_value" ]]; then
      win_root="$(_to_unix "$env_value")"
      break
    fi
  done
  if [[ -n "$win_root" ]]; then
    win_cli="$win_root/VOXPrint AI Suite/Enlarge/voxprint-enlarge-cli.exe"
  fi
  if [[ -n "${LOCALAPPDATA:-}" ]]; then
    legacy="$(_to_unix "$LOCALAPPDATA")/Voxprint/Enlarge/VoxprintEnlarge.exe"
  fi
  if command -v voxprint-enlarge-cli >/dev/null 2>&1; then
    APP=(voxprint-enlarge-cli)
  elif command -v voxprint-enlarge >/dev/null 2>&1; then
    APP=(voxprint-enlarge)
  elif [[ -x /opt/voxprint-ai-suite/enlarge/voxprint-enlarge ]]; then
    APP=(/opt/voxprint-ai-suite/enlarge/voxprint-enlarge)
  elif [[ -n "$win_cli" && -f "$win_cli" ]]; then
    APP=("$win_cli")
  elif [[ -n "$legacy" && -f "$legacy" ]]; then
    APP=("$legacy")
  elif [[ -f "$ROOT/main.py" ]]; then
    APP=("${PYTHON[@]}" "$ROOT/main.py")
  else
    echo "setup: enlarge is not installed and main.py is not next to scripts/" >&2
    exit 2
  fi

  SAMPLE="${VOXPRINT_VM_SAMPLE:-$ROOT/samples/sample.png}"
  LINE="${VOXPRINT_VM_LINE:-$SAMPLE}"
  PHOTO="${VOXPRINT_VM_PHOTO:-$SAMPLE}"
  local needed
  for needed in "$SAMPLE" "$LINE" "$PHOTO"; do
    if [[ ! -f "$needed" ]]; then
      echo "setup: missing image $needed" >&2
      exit 2
    fi
  done

  fail() { echo "fail: $*" >&2; exit 1; }
  setup() { echo "setup: $*" >&2; exit 2; }

  run_json() {
    local name="$1"
    shift
    local out="$OUT/$name.out"
    local err="$OUT/$name.err"
    "${APP[@]}" "$@" >"$out" 2>"$err"
    local rc=$?
    printf '%s' "$rc"
  }

  local caps gpu_status selftest stock cached anime chain chain_rc still holder
  caps="$(run_json caps --capabilities --json)"
  if [[ "$caps" -ne 0 ]]; then
    setup "capabilities exited $caps"
  fi
  gpu_status="$("${PYTHON[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["gpu"]["status"])' "$OUT/caps.out")" || setup "capabilities JSON"
  if [[ "$gpu_status" == "none" ]]; then
    echo "no usable GPU" >&2
    exit 3
  fi

  selftest="$(run_json selftest --selftest --json)"
  if [[ "$selftest" -ne 0 ]]; then
    fail "selftest exited $selftest"
  fi
  "${PYTHON[@]}" - "$OUT/selftest.out" <<'PY' || fail "selftest JSON"
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if not data.get("ok"):
    raise SystemExit("ok is not true")
for name in ("tensorrt_ep", "cuda_ep"):
    item = data.get(name) or {}
    if item.get("detail") != "loaded":
        raise SystemExit(f"{name} is {item.get('detail')!r}")
PY

  check_progress() {
    local file="$1"
    local mode="${2:-order}"
    "${PYTHON[@]}" - "$file" "$mode" <<'PY' || return 1
import json, sys
path, mode = sys.argv[1], sys.argv[2]
order = ["queued", "waiting_gpu", "loading", "building_engine", "loading_cached_engine", "tiles", "encoding"]
rank = {name: index for index, name in enumerate(order)}
last = -1
engine = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    message = json.loads(line)
    if message.get("op") != "progress":
        continue
    stage = str(message.get("stage") or "")
    if stage not in rank:
        raise SystemExit(f"unknown stage {stage}")
    if rank[stage] < last:
        raise SystemExit(f"stage went backwards to {stage}")
    last = rank[stage]
    if stage in {"building_engine", "loading_cached_engine"}:
        engine.append(stage)
if mode == "cached" and (not engine or engine[0] != "loading_cached_engine"):
    raise SystemExit(f"cached run reported {engine[0] if engine else 'no engine stage'}")
if mode == "build" and "building_engine" not in engine:
    raise SystemExit("first run did not report building_engine")
if mode not in {"order", "cached", "build"}:
    raise SystemExit(f"unknown progress check {mode}")
PY
  }

  stock="$(run_json stock upscale --input "$SAMPLE" --output-dir "$OUT" --scale 2 --format avif --json-progress)"
  if [[ "$stock" -ne 0 ]]; then
    if grep -q '"code": "model_missing"\|"code":"model_missing"' "$OUT/stock.out" 2>/dev/null; then
      setup "default model is not installed"
    fi
    fail "stock upscale exited $stock"
  fi
  check_progress "$OUT/stock.out" build || fail "stock run did not report building_engine"
  find "$OUT" -maxdepth 1 -name '*.enlarged-2x.avif' | grep -q . || fail "stock upscale wrote no avif"

  cached="$(run_json stock-cached upscale --input "$SAMPLE" --output-dir "$OUT" --scale 2 --format avif --json-progress)"
  if [[ "$cached" -ne 0 ]]; then
    fail "cached upscale exited $cached"
  fi
  check_progress "$OUT/stock-cached.out" cached || fail "cached run did not report loading_cached_engine"

  anime="$(run_json anime upscale --input "$LINE" --output-dir "$OUT" --scale 4 --preset art --format avif --json-progress)"
  if [[ "$anime" -ne 0 ]]; then
    if grep -q 'model_missing' "$OUT/anime.out" 2>/dev/null; then
      setup "realesrgan-x4plus-anime6b is not installed"
    fi
    fail "anime6b upscale exited $anime"
  fi
  check_progress "$OUT/anime.out" order || fail "anime progress went backwards"

  LOCK="$OUT/voxprint-gpu.lock"
  export VOXPRINT_GPU_LOCK="$LOCK"
  holder=""
  trap '[[ -n "${holder}" ]] && kill "$holder" 2>/dev/null || true' EXIT
  sleep 3600 &
  holder=$!
  "${PYTHON[@]}" - "$LOCK" "$holder" <<'PY' || setup "could not write the parent lock"
import json, sys
from datetime import datetime, timedelta, timezone
now = datetime.now(timezone.utc)
eta = now + timedelta(hours=1)
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps({
    "owner": "canvas",
    "pid": int(sys.argv[2]),
    "job": "vm-test",
    "started": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "eta": eta.strftime("%Y-%m-%dT%H:%M:%SZ"),
}))
PY
  chain="$(run_json chain upscale --input "$PHOTO" --output "$OUT/chain.avif" --long-side 3840 --format avif --json-progress --parent-lock-pid "$holder")"
  chain_rc="$chain"
  still="$("${PYTHON[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("pid"))' "$LOCK" 2>/dev/null || echo gone)"
  kill "$holder" 2>/dev/null || true
  wait "$holder" 2>/dev/null || true
  unset VOXPRINT_GPU_LOCK
  if [[ "$chain_rc" -ne 0 ]]; then
    fail "long-side chain exited $chain_rc"
  fi
  if [[ "$still" != "$holder" ]]; then
    fail "parent lock pid changed from $holder to $still"
  fi
  [[ -f "$OUT/chain.avif" ]] || fail "chain wrote no file"
  check_progress "$OUT/chain.out" order || fail "chain progress went backwards"

  echo "smoke passed ($OUT)"
  exit 0
}

# `return` succeeds only while this file is sourced. Executing it runs the smoke test.
if ! (return 0 2>/dev/null); then
  vm_test_main "$@"
fi
