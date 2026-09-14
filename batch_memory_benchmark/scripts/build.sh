#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${BENCHMARK_DIR}/../set_env_claix23_cuda12.4.sh" ]]; then
    source "${BENCHMARK_DIR}/../set_env_claix23_cuda12.4.sh"
fi

echo "=== Building Batch Memory Benchmark ==="
cd "${BENCHMARK_DIR}"

mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16

echo "=== Build Complete: ${BENCHMARK_DIR}/build/fake_solver ==="
