#!/usr/bin/env bash
#SBATCH --job-name=artifact_analysis
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
# Runtime estimate from login-node timing with the precision-family metrics:
# NCU ~1 min/point for CNN/MLPs and 4-6 min for the transformer, so the x8
# ladders (~30 points) take ~45-65 min; measurement ~15-20 min, Nsight Systems
# ~8-10 min, report ~1 min. Total ~70-95 min under a 2 h allocation.
#SBATCH --time=02:00:00
#SBATCH --output=generated/slurm_%j.out
#SBATCH --error=generated/slurm_%j.err

# Full campaign: adaptive measurement + NCU (warm, range replay) + Nsight
# Systems + tables + figures + LaTeX. Submit with:
#   sbatch --exclusive run_analysis.sbatch
# Environment overrides:
#   CATALOG (default model_catalog.json; use model_catalog_replicas.json for
#   auto-generated replicas), MODELS, PRECISIONS, MEMORY_FRACTION, LADDER
#   (x<N>, default x8), MAX_BATCH, BATCH_LADDER,
#   COMPARISON_BATCH, NCU_BIN, NCU_BATCHES (default ladder; use e.g. 1,1000 to
#   cap), NCU_MAX_BATCH, NCU_CACHE_CONTROL, NCU_REPLAY_MODE, NCU_CLOCK_CONTROL,
#   NSYS_BIN, SKIP_NCU=1, SKIP_NSYS=1, NO_LATEX=1, NO_COMPILE_LATEX=1,
#   LATEXMK, TEXLIVE_BIN, EXTRA_ARGS

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
ENV_SCRIPT="${SCRIPT_DIR}/set_env_claix23_cuda12.4.sh"
if [[ ! -f "${ENV_SCRIPT}" && -f "${SCRIPT_DIR}/../set_env_claix23_cuda12.4.sh" ]]; then
    ENV_SCRIPT="${SCRIPT_DIR}/../set_env_claix23_cuda12.4.sh"
fi
if [[ -f "${ENV_SCRIPT}" ]]; then
    # The environment script is not fully compatible with `set -u` (some
    # EBROOT* variables may be unset on newer partitions), so relax it here.
    set +u
    # shellcheck disable=SC1091
    source "${ENV_SCRIPT}"
    set -u
fi
mkdir -p generated

ARGS=(
    --catalog "${CATALOG:-${SCRIPT_DIR}/model_catalog.json}"
    --output-dir generated
    --run-name "slurm_${SLURM_JOB_ID}"
    --models "${MODELS:-watercnn,mmcp_test_mlp_m5,tbl_transformer,giant_mlp}"
    --precisions "${PRECISIONS:-fp32}"
    --memory-fraction "${MEMORY_FRACTION:-0.9}"
    --ladder "${LADDER:-x8}"
    --comparison-batch "${COMPARISON_BATCH:-1000}"
    --ncu-bin "${NCU_BIN:-/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.8.0/bin/ncu}"
    --ncu-batches "${NCU_BATCHES:-ladder}"
    --ncu-cache-control "${NCU_CACHE_CONTROL:-none}"
    --ncu-replay-mode "${NCU_REPLAY_MODE:-application}"
    --ncu-clock-control "${NCU_CLOCK_CONTROL:-none}"
    --nsys-bin "${NSYS_BIN:-nsys}"
)
[[ -n "${NCU_MAX_BATCH:-}" ]] && ARGS+=(--ncu-max-batch "${NCU_MAX_BATCH}")
[[ -n "${LATEXMK:-}" ]] && ARGS+=(--latexmk "${LATEXMK}")
[[ -n "${TEXLIVE_BIN:-}" ]] && ARGS+=(--texlive-bin "${TEXLIVE_BIN}")
[[ -n "${MAX_BATCH:-}" ]] && ARGS+=(--max-batch "${MAX_BATCH}")
[[ -n "${BATCH_LADDER:-}" ]] && ARGS+=(--batch-ladder "${BATCH_LADDER}")
[[ "${SKIP_NCU:-0}" == "1" ]] && ARGS+=(--skip-ncu)
[[ "${SKIP_NSYS:-0}" == "1" ]] && ARGS+=(--skip-nsys)
[[ "${NO_LATEX:-0}" == "1" ]] && ARGS+=(--no-latex)
[[ "${NO_COMPILE_LATEX:-0}" == "1" ]] && ARGS+=(--no-compile-latex)

python analyze.py "${ARGS[@]}" ${EXTRA_ARGS:-}
