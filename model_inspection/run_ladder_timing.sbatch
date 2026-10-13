#!/usr/bin/env bash
#SBATCH --job-name=ladder_timing
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --exclusive
# Timing-only x2 ladder campaign for the thesis throughput figure. No NCU and
# no Nsight Systems: the measure stage alone records CUDA-event timings per
# batch (1 VRAM probe + 10 warm-up + 30 timed + 30 resident forwards per
# point, medians over the 30 timed iterations). One entry per invocation so
# each gets its own batch ladder and run directory; entries run sequentially
# on one GPU. Giant MLP (B up to 1.5e6) dominates at ~95-100 min.
#
# Host memory: PyTorch's caching host allocator never returns freed pinned
# blocks to the OS, so a single process accumulates the per-batch pinned
# input+output buffers (~14 KiB/sample for the MLP). The MLP x2 grid sums to
# ~60 GiB for B<=2e6 plus ~57 GiB for B=4e6; keeping both in one process
# exceeds the 122 GB/GPU cap, so the MLP ladder is split across two
# invocations (mlp_a, mlp_b), each with a fresh process and pinned cache.
#SBATCH --time=03:00:00
#SBATCH --output=generated/slurm_%j.out
#SBATCH --error=generated/slurm_%j.err
#
# Submit with:
#   sbatch run_ladder_timing.sbatch
# The ladder for each model is a x2 grid from B=1 up to the largest power of
# two at or below the adopted maximum batch, with that maximum appended
# (a non-power-of-two endpoint). Environment overrides:
#   CATALOG, WARMUP (default 10), ITERATIONS (default 30)

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
ENV_SCRIPT="${SCRIPT_DIR}/set_env_claix23_cuda12.4.sh"
if [[ ! -f "${ENV_SCRIPT}" && -f "${SCRIPT_DIR}/../set_env_claix23_cuda12.4.sh" ]]; then
    ENV_SCRIPT="${SCRIPT_DIR}/../set_env_claix23_cuda12.4.sh"
fi
if [[ -f "${ENV_SCRIPT}" ]]; then
    # The environment script is not fully compatible with `set -u`.
    set +u
    # shellcheck disable=SC1091
    source "${ENV_SCRIPT}"
    set -u
fi
mkdir -p generated

declare -A LADDER=(
    [watercnn]="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,110000000"
    [mlp_a]="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152"
    [mlp_b]="4000000"
    [tbl_transformer]="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65000"
    [giant_mlp]="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,1500000"
)
declare -A MODEL=(
    [watercnn]=watercnn
    [mlp_a]=mmcp_test_mlp_m5
    [mlp_b]=mmcp_test_mlp_m5
    [tbl_transformer]=tbl_transformer
    [giant_mlp]=giant_mlp
)
ORDER=(watercnn mlp_a mlp_b tbl_transformer giant_mlp)

RUN_ROOT="generated/slurm_${SLURM_JOB_ID}"
for entry in "${ORDER[@]}"; do
    echo "=== $(date -Is) measuring ${MODEL[$entry]} as ${entry} (ladder: ${LADDER[$entry]})"
    python analyze.py \
        --catalog "${CATALOG:-${SCRIPT_DIR}/model_catalog.json}" \
        --output-dir "${RUN_ROOT}" \
        --run-name "${entry}" \
        --models "${MODEL[$entry]}" \
        --precisions fp32 \
        --warmup "${WARMUP:-10}" \
        --iterations "${ITERATIONS:-30}" \
        --batch-ladder "${LADDER[$entry]}" \
        --stage measure
done
echo "=== $(date -Is) ladder timing campaign complete: ${RUN_ROOT}"
