#!/bin/bash
# Compare CPU vs GPU A-matrix build for a chimes_lsq test case.
# Requires a CUDA build: export hosttype=UT-TACC-GPU; ./install.sh 0 "" 1 1 1
# See doc/GPU_ACCELERATION.md for full GPU documentation and tech debt.
# Usage: ./scripts/gpu_validate.sh [path/to/test/dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE_DIR="${1:-${ROOT}/test_suite-lsq/special3b}"
FM_SETUP="${CASE_DIR}/fm_setup.in"
LSQ="${ROOT}/build/chimes_lsq"

if [[ ! -x "${LSQ}" ]]; then
    echo "ERROR: build chimes_lsq first:" >&2
    echo "  export hosttype=UT-TACC-GPU; ./install.sh 0 \"\" 1 1 1" >&2
    exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
cp "${FM_SETUP}" "${WORKDIR}/"
cp "${CASE_DIR}"/*.xyzf "${WORKDIR}/" 2>/dev/null || true
cd "${WORKDIR}"

run_build() {
    local tag="$1"
    shift
    rm -f A.*.txt b.*.txt dim.txt
    env "$@" "${LSQ}" fm_setup.in > "${tag}.log" 2>&1
    mv A.0000.txt "${tag}_A.txt"
    mv b.0000.txt "${tag}_b.txt"
}

echo "=== CPU reference ==="
run_build cpu CHIMES_LSQ_USE_GPU=0

echo "=== GPU build ==="
run_build gpu CHIMES_LSQ_USE_GPU=1

python3 - <<'PY'
import numpy as np, sys

def load(path):
    return np.genfromtxt(path, dtype=float)

for name in ('A', 'b'):
    cpu = load(f'cpu_{name}.txt')
    gpu = load(f'gpu_{name}.txt')
    if cpu.shape != gpu.shape:
        print(f"FAIL: {name} shape {cpu.shape} vs {gpu.shape}", file=sys.stderr)
        sys.exit(1)
    d = np.max(np.abs(cpu - gpu))
    print(f"{name}: max abs diff = {d:.6e}")
    if d > 1e-10:
        print(f"FAIL: {name} mismatch", file=sys.stderr)
        sys.exit(1)
print("PASS: GPU A/b matches CPU")
PY

echo "Validation complete."
