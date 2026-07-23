#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PYBIND11_DIR="$("$PYTHON_BIN" -m pybind11 --cmakedir)"

echo "Python:   $PYTHON_BIN"
echo "pybind11: $PYBIND11_DIR"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DPython3_EXECUTABLE="$PYTHON_BIN"
    -Dpybind11_DIR="$PYBIND11_DIR"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
    )
fi

cmake -S . -B build "${CMAKE_ARGS[@]}"

cmake --build build -j"$(nproc)"
