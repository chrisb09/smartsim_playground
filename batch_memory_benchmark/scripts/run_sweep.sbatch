#!/usr/bin/env bash
#SBATCH --job-name=batch_mem_sweep
#SBATCH --account=thes2181
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:30:00
#SBATCH --output=logs/batch_mem_sweep_%j.out

set -euo pipefail

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${BENCHMARK_DIR}/scripts"
cd "${BENCHMARK_DIR}"

if [[ -f "${BENCHMARK_DIR}/../set_env_claix23_cuda12.4.sh" ]]; then
    source "${BENCHMARK_DIR}/../set_env_claix23_cuda12.4.sh"
fi

mkdir -p logs

echo "================================================================="
echo "=== Batch & Memory Calibration Sweep on 1 GPU (${SLURM_JOB_ID}) ==="
echo "================================================================="

# Build first if not present
"${SCRIPT_DIR}/build.sh"

BATCH_SIZES=(100000 500000 2000000)
PROVIDERS=(smartsim aix_collective aix_pipelined phydll_cpp phydll_py)

for prov in "${PROVIDERS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        run_id="${SLURM_JOB_ID}_${prov}_b${bs}"
        echo ""
        echo ">>> Running Provider: ${prov} | Batch Size: ${bs} <<<"
        "${SCRIPT_DIR}/run_case.sh" "${prov}" "${bs}" 100000 12 2 "${run_id}" || {
            echo "WARNING: Case ${prov} b=${bs} failed with exit code $?"
        }
        sleep 2
    done
done

echo ""
echo "================================================================="
echo "=== Sweep Completed. Running Linear Regression Fitting...     ==="
echo "================================================================="

python3 "${SCRIPT_DIR}/fit_vram_and_cache_limits.py" "logs"

echo "=== All Done ==="
