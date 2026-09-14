#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${1:-gpu_telemetry.csv}"
INTERVAL_MS="${2:-50}"

echo "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free" > "${OUTPUT_FILE}"

exec nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.free \
    --format=csv,noheader,nounits \
    -lms "${INTERVAL_MS}" >> "${OUTPUT_FILE}" 2>/dev/null
