#!/usr/bin/env bash
#SBATCH --job-name=aix_coll_smoke
#SBATCH --account=thes2181
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aix_coll_smoke_%j.out

set -euo pipefail

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${BENCHMARK_DIR}"

source "${BENCHMARK_DIR}/../set_env_claix23_cuda12.4.sh"

mkdir -p logs

echo "=== AIx Collective B=2M smoke (post AIX_ASYNC_GPU_PIPELINE removal) ==="
"${BENCHMARK_DIR}/scripts/run_case.sh" aix_collective 2000000 100000 12 2 "smoke_${SLURM_JOB_ID}"

echo "=== Done ==="
