#!/bin/bash
#SBATCH -J h100_hw_peaks
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH -p c23g
#SBATCH -A rwth0792
#SBATCH -o logs/h100_peaks_%j.out

set -euo pipefail

WORKDIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/h100_nvl_94gb_identification"
cd "$WORKDIR"
mkdir -p logs reports

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"

echo "================================================================="
echo "=== NVIDIA Hopper H100 Hardware Peak Benchmark ==="
echo "================================================================="
echo "Host: $(hostname -f)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Account: ${SLURM_JOB_ACCOUNT}"
date
echo ""

echo "--- GPU Status & Power Limits ---"
nvidia-smi
echo ""
nvidia-smi topo -m
echo ""

REPORT_FILE="${WORKDIR}/reports/h100_peaks_$(hostname -s)_job${SLURM_JOB_ID}.json"

echo "Running hardware peaks benchmark (FP32, TF32, HBM3, NVLink all-pairs)..."
"$PYTHON_BIN" "$WORKDIR/benchmark_hardware_peaks.py" \
    --gpu 0 \
    --output "$REPORT_FILE"

echo ""
echo "=== Benchmark Completed Successfully ==="
date
