#!/usr/bin/env bash
#SBATCH --job-name=measure_stats
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
# Runtime estimate: the NCU pass dominates. Five repetitions at two batches
# for four models are ~40 NCU invocations (~1-1.5 h); ten repetitions double
# that. Timing adds only minutes, plus the VRAM probe/verification forwards.
#SBATCH --time=04:00:00
#SBATCH --output=generated/slurm_stats_%j.out
#SBATCH --error=generated/slurm_stats_%j.err

# Repeated measurement statistics at B=1 and ~80% VRAM (CUDA-event timing plus
# NCU counters), reported as median/mean/std/min/max/IQR/CV. Submit with:
#   sbatch --exclusive run_stats.sbatch
# Flags after the script name are forwarded to measure_stats.py and override the
# environment defaults (useful when the batch environment does not propagate
# submit-host variables), for example:
#   sbatch --exclusive run_stats.sbatch --run-name <existing-run> --resume \
#       --models giant_mlp --warmup 5 --iterations 10 --repeats 5
# Environment overrides:
#   CATALOG (default model_catalog.json; use model_catalog_replicas.json for
#   auto-generated replicas), MODELS, PRECISIONS, REPEATS (default 5),
#   MEMORY_FRACTION (default 0.8), LADDER, MAX_BATCH, FIT_FROM_RUN, PROBE=1,
#   SKIP_NCU=1, NCU_BIN, NCU_TIMEOUT, NCU_CACHE_CONTROL, NCU_REPLAY_MODE,
#   NCU_CLOCK_CONTROL, WARMUP, ITERATIONS, EXTRA_ARGS

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
ENV_SCRIPT="${SCRIPT_DIR}/set_env_claix23_cuda12.4.sh"
if [[ ! -f "${ENV_SCRIPT}" && -f "${SCRIPT_DIR}/../set_env_claix23_cuda12.4.sh" ]]; then
    ENV_SCRIPT="${SCRIPT_DIR}/../set_env_claix23_cuda12.4.sh"
fi
if [[ -f "${ENV_SCRIPT}" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${ENV_SCRIPT}"
    set -u
fi
mkdir -p generated

ARGS=(
    --catalog "${CATALOG:-${SCRIPT_DIR}/model_catalog.json}"
    --models "${MODELS:-watercnn,mmcp_test_mlp_m5,tbl_transformer,giant_mlp}"
    --precisions "${PRECISIONS:-fp32}"
    --repeats "${REPEATS:-5}"
    --warmup "${WARMUP:-10}"
    --iterations "${ITERATIONS:-30}"
    --memory-fraction "${MEMORY_FRACTION:-0.8}"
    --ladder "${LADDER:-x8}"
    --output-dir generated
    --run-name "slurm_stats_${SLURM_JOB_ID}"
    --ncu-bin "${NCU_BIN:-/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.8.0/bin/ncu}"
    --ncu-timeout "${NCU_TIMEOUT:-1800}"
    --ncu-cache-control "${NCU_CACHE_CONTROL:-none}"
    --ncu-replay-mode "${NCU_REPLAY_MODE:-application}"
    --ncu-clock-control "${NCU_CLOCK_CONTROL:-none}"
)
[[ -n "${MAX_BATCH:-}" ]] && ARGS+=(--max-batch "${MAX_BATCH}")
[[ -n "${FIT_FROM_RUN:-}" ]] && ARGS+=(--fit-from-run "${FIT_FROM_RUN}")
[[ "${PROBE:-0}" == "1" ]] && ARGS+=(--probe)
[[ "${SKIP_NCU:-0}" == "1" ]] && ARGS+=(--skip-ncu)

python measure_stats.py "${ARGS[@]}" "$@" ${EXTRA_ARGS:-}
