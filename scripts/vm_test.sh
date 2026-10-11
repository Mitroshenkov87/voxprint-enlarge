#!/usr/bin/env bash
# Non-interactive GPU checks for a Voxprint AI Enlarge machine.
# Exit 0 pass, 1 a test failed, 2 setup, 3 no usable GPU.
# --smoke is the short set (selftest, one stock upscale and its cached rerun,
# anime 6B at 4x, and the long-side chain). It is also the default.
#
# The interpreter is the Voxprint shared runtime python, then `py -3`, then
# `python3`. On Windows `python3` is often the Microsoft Store alias, so it
# is last. The chosen command is printed as `python: ...`.
set -u

# Print one interpreter command. A path is one line. `py -3` is the launcher.
resolve_vm_python() {
  local candidate data
  if [[ -n "${VOXPRINT_HOME:-}" ]]; then
    for candidate in "$VOXPRINT_HOME/shared/python/python.exe" "$VOXPRINT_HOME/shared/python/bin/python3"; do
      if [[ -f "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi
  if [[ -n "${LOCALAPPDATA:-}" ]]; then
    candidate="$LOCALAPPDATA/Voxprint/shared/python/python.exe"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  data="${XDG_DATA_HOME:-${HOME:-}/.local/share}"
  candidate="$data/voxprint/shared/python/bin/python3"
  if [[ -f "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi
  if command -v py >/dev/null 2>&1 && py -3 -c "import sys" >/dev/null 2>&1; then
    printf '%s\n' "py -3"
    return 0
  fi
  printf '%s\n' "python3"
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
  choice="$(resolve_vm_python)"
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
  if command -v voxprint-enlarge >/dev/null 2>&1; then
    APP=(voxprint-enlarge)
  elif [[ -n "${LOCALAPPDATA:-}" && -x "$LOCALAPPDATA/Voxprint/Enlarge/VoxprintEnlarge.exe" ]]; then
    APP=("$LOCALAPPDATA/Voxprint/Enlarge/VoxprintEnlarge.exe")
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
    local cached="${2:-0}"
    "${PYTHON[@]}" - "$file" "$cached" <<'PY' || return 1
import json, sys
path, cached = sys.argv[1], sys.argv[2] == "1"
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
if cached and engine and engine[0] != "loading_cached_engine":
    raise SystemExit(f"cached run reported {engine[0]}")
PY
  }

  stock="$(run_json stock upscale --input "$SAMPLE" --output-dir "$OUT" --scale 2 --format avif --json-progress)"
  if [[ "$stock" -ne 0 ]]; then
    if grep -q '"code": "model_missing"\|"code":"model_missing"' "$OUT/stock.out" 2>/dev/null; then
      setup "default model is not installed"
    fi
    fail "stock upscale exited $stock"
  fi
  check_progress "$OUT/stock.out" 0 || fail "stock progress went backwards"
  find "$OUT" -maxdepth 1 -name '*.enlarged-2x.avif' | grep -q . || fail "stock upscale wrote no avif"

  cached="$(run_json stock-cached upscale --input "$SAMPLE" --output-dir "$OUT" --scale 2 --format avif --json-progress)"
  if [[ "$cached" -ne 0 ]]; then
    fail "cached upscale exited $cached"
  fi
  check_progress "$OUT/stock-cached.out" 1 || fail "cached run did not report loading_cached_engine"

  anime="$(run_json anime upscale --input "$LINE" --output-dir "$OUT" --scale 4 --preset art --format avif --json-progress)"
  if [[ "$anime" -ne 0 ]]; then
    if grep -q 'model_missing' "$OUT/anime.out" 2>/dev/null; then
      setup "realesrgan-x4plus-anime6b is not installed"
    fi
    fail "anime6b upscale exited $anime"
  fi
  check_progress "$OUT/anime.out" 0 || fail "anime progress went backwards"

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
  check_progress "$OUT/chain.out" 0 || fail "chain progress went backwards"

  echo "smoke passed ($OUT)"
  exit 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  vm_test_main "$@"
fi
