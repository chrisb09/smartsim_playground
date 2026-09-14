#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BENCHMARK_DIR}"

PROVIDER="${1:-smartsim}" # smartsim, aix_collective, aix_pipelined, phydll_cpp, phydll_py
BATCH_SIZE="${2:-100000}"
SAMPLES_PER_RANK="${3:-100000}"
STEPS="${4:-12}"
WARMUP_STEPS="${5:-2}"
RUN_ID="${6:-$(date +%s)}"

MODEL_PATH="${BENCHMARK_DIR}/../mini_app/train_models/model_a/watercnn_cuda.pt"
LOG_DIR="${BENCHMARK_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_PREFIX="${LOG_DIR}/bench_${PROVIDER}_b${BATCH_SIZE}_${RUN_ID}"
SOLVER_LOG="${LOG_PREFIX}.log"
SIDECAR_CSV="${LOG_PREFIX}_gpu_telemetry.csv"
STATS_CSV="${LOG_DIR}/benchmark_results.csv"

echo "================================================================="
echo "=== Running Case: ${PROVIDER} | Batch: ${BATCH_SIZE} | ID: ${RUN_ID} ==="
echo "================================================================="

# Start GPU memory telemetry sidecar in background
"${SCRIPT_DIR}/gpu_memory_sidecar.sh" "${SIDECAR_CSV}" 50 &
SIDECAR_PID=$!
trap 'kill ${SIDECAR_PID} 2>/dev/null || true' EXIT

sleep 1 # Let sidecar capture baseline idle VRAM

SOLVER_EXE="${BENCHMARK_DIR}/build/fake_solver"
if [[ ! -x "${SOLVER_EXE}" ]]; then
    echo "ERROR: fake_solver binary not found. Run scripts/build.sh first." >&2
    exit 1
fi

case "${PROVIDER}" in
    smartsim)
        # Start local SmartSim / RedisAI database on available port
        DB_PORT=$(( 6380 + (RANDOM % 1000) ))
        echo "Starting SmartSim Redis DB on port ${DB_PORT}..."
        
        PYTHON_EXE="${BENCHMARK_DIR}/../python/smartsim_cuda-12/bin/python"
        
        # Start SmartSim DB via Python controller script
        cat << PY > "${LOG_DIR}/start_db_${RUN_ID}.py"
import os, sys, time
from smartsim.experiment import Experiment
exp = Experiment("db_${RUN_ID}", launcher="local")
db = exp.create_database(port=${DB_PORT}, interface="lo", db_nodes=1, threads_per_queue=4, intra_op_threads=1, inter_op_threads=1)
exp.generate(db, overwrite=True)
exp.start(db)
print(f"DB_READY:127.0.0.1:{${DB_PORT}}")
sys.stdout.flush()
time.sleep(600)
PY
        "${PYTHON_EXE}" "${LOG_DIR}/start_db_${RUN_ID}.py" > "${LOG_PREFIX}_db.log" 2>&1 &
        DB_PID=$!
        
        # Wait for DB ready
        export SSDB="127.0.0.1:${DB_PORT}"
        export SR_MODEL_TIMEOUT=900000
        export SR_CMD_TIMEOUT=900000
        export SR_SOCKET_TIMEOUT=900000
        for i in {1..30}; do
            if grep -q "DB_READY" "${LOG_PREFIX}_db.log" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        # Run fake solver with 20 MPI ranks (each 100k samples = 2M total)
        # Configure min_batch_size = batch_size to test controlled aggregation
        mpirun --bind-to none -x SSDB -x SR_MODEL_TIMEOUT -x SR_CMD_TIMEOUT -x SR_SOCKET_TIMEOUT -n 20 \
            "${SOLVER_EXE}" \
            --config "${BENCHMARK_DIR}/configs/smartsim.toml" \
            --provider smartsim \
            --model-path "${MODEL_PATH}" \
            --samples-per-rank "${SAMPLES_PER_RANK}" \
            --steps "${STEPS}" \
            --warmup-steps "${WARMUP_STEPS}" \
            --batch-size "${BATCH_SIZE}" \
            --min-batch-size "${BATCH_SIZE}" \
            --min-batch-timeout 1000 \
            --output-csv "${STATS_CSV}" \
            2>&1 | tee "${SOLVER_LOG}"

        # Cleanly terminate DB
        kill ${DB_PID} 2>/dev/null || true
        rm -f "${LOG_DIR}/start_db_${RUN_ID}.py"
        ;;

    aix_collective)
        # AIx Collective: 20 total ranks (19 CPU + 1 GPU controller)
        mpirun --bind-to none -n 20 \
            "${SOLVER_EXE}" \
            --config "${BENCHMARK_DIR}/configs/aix.toml" \
            --provider aix \
            --aix-comm-mode collective \
            --model-path "${MODEL_PATH}" \
            --samples-per-rank "${SAMPLES_PER_RANK}" \
            --steps "${STEPS}" \
            --warmup-steps "${WARMUP_STEPS}" \
            --batch-size "${BATCH_SIZE}" \
            --output-csv "${STATS_CSV}" \
            2>&1 | tee "${SOLVER_LOG}"
        ;;

    aix_pipelined)
        # AIx Pipelined: 20 total ranks (19 CPU + 1 GPU controller)
        mpirun --bind-to none -n 20 \
            "${SOLVER_EXE}" \
            --config "${BENCHMARK_DIR}/configs/aix.toml" \
            --provider aix \
            --aix-comm-mode pipelined \
            --model-path "${MODEL_PATH}" \
            --samples-per-rank "${SAMPLES_PER_RANK}" \
            --steps "${STEPS}" \
            --warmup-steps "${WARMUP_STEPS}" \
            --batch-size "${BATCH_SIZE}" \
            --output-csv "${STATS_CSV}" \
            2>&1 | tee "${SOLVER_LOG}"
        ;;

    phydll_cpp)
        # PhyDLL C++ Uniform: 20 solver ranks + 1 DL client rank (MPMD)
        DL_CLIENT="${BENCHMARK_DIR}/../CPP-ML-Interface/dl_clients/build-miniapp/phydll_dl_client"
        PHYDLL_LIB="${BENCHMARK_DIR}/../CPP-ML-Interface/extern/phydll/build/lib"
        export PHYDLL_MPMD_SHUTDOWN_BARRIER=1
        export PHYDLL_DL_FIELD_COUNT=1
        
        mpirun --bind-to none \
            -x LD_LIBRARY_PATH="${PHYDLL_LIB}:${LD_LIBRARY_PATH:-}" \
            -x PHYDLL_DL_FIELD_COUNT \
            -n 20 "${SOLVER_EXE}" \
                --config "${BENCHMARK_DIR}/configs/phydll_uniform.toml" \
                --provider phydll \
                --model-path "${MODEL_PATH}" \
                --samples-per-rank "${SAMPLES_PER_RANK}" \
                --steps "${STEPS}" \
                --warmup-steps "${WARMUP_STEPS}" \
                --batch-size "${BATCH_SIZE}" \
                --output-csv "${STATS_CSV}" \
            : -n 1 -x LD_LIBRARY_PATH="${PHYDLL_LIB}:${LD_LIBRARY_PATH:-}" -x PHYDLL_DL_FIELD_COUNT "${DL_CLIENT}" \
            2>&1 | tee "${SOLVER_LOG}"
        ;;

    phydll_py)
        # PhyDLL Python Uniform: 20 solver ranks + 1 Python DL client rank
        DL_PY="${BENCHMARK_DIR}/../CPP-ML-Interface/dl_clients/phydll_dl_client.py"
        PHYDLL_LIB="${BENCHMARK_DIR}/../CPP-ML-Interface/extern/phydll/build/lib"
        PYTHON_EXE="${BENCHMARK_DIR}/../python/smartsim_cuda-12/bin/python"
        export PHYDLL_MPMD_SHUTDOWN_BARRIER=1
        export PHYDLL_DL_FIELD_COUNT=1

        mpirun --bind-to none \
            -x LD_LIBRARY_PATH="${PHYDLL_LIB}:${LD_LIBRARY_PATH:-}" \
            -x PHYDLL_DL_FIELD_COUNT \
            -n 20 "${SOLVER_EXE}" \
                --config "${BENCHMARK_DIR}/configs/phydll_uniform.toml" \
                --provider phydll \
                --model-path "${MODEL_PATH}" \
                --samples-per-rank "${SAMPLES_PER_RANK}" \
                --steps "${STEPS}" \
                --warmup-steps "${WARMUP_STEPS}" \
                --batch-size "${BATCH_SIZE}" \
                --output-csv "${STATS_CSV}" \
            : -n 1 -x LD_LIBRARY_PATH="${PHYDLL_LIB}:${LD_LIBRARY_PATH:-}" -x PHYDLL_DL_FIELD_COUNT "${PYTHON_EXE}" "${DL_PY}" \
            2>&1 | tee "${SOLVER_LOG}"
        ;;

    *)
        echo "Unknown provider: ${PROVIDER}" >&2
        exit 1
        ;;
esac

sleep 1 # Let sidecar capture final memory state
kill ${SIDECAR_PID} 2>/dev/null || true

echo "=== Finished ${PROVIDER} (b=${BATCH_SIZE}). Log: ${SOLVER_LOG} | Telemetry: ${SIDECAR_CSV} ==="
