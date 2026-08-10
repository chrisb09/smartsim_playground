#!/bin/bash
# model_inspection/run_inspection.sbatch
# =======================================
# Slurm job to run the model inspection suite on a GPU node (H100 / A100).
#
# Adjust #SBATCH directives to match your queue/partition/account.
#
# Usage:
#   sbatch run_inspection.sbatch
#   sbatch run_inspection.sbatch --export=SKIP_GIANT=1    (skip 1.56 GiB giant MLP)
#   sbatch run_inspection.sbatch --export=MODELS=watercnn,transformer,mmcp_transformer_core
#
# Results are written to:
#   model_inspection/results/inspection_<timestamp>_<device>.json
#   model_inspection/results/inspection_<timestamp>_<device>.md
#   model_inspection/results/inspection_<timestamp>_cuda_roofline_coupling.{pdf,png}
#   model_inspection/results/inspection_<timestamp>_cuda_roofline_device.{pdf,png}
#   model_inspection/results/inspection_<timestamp>_cuda_roofline_combined.{pdf,png}
#
#SBATCH --job-name=model_inspect
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
# Uncomment and set your account/reservation if required:
# #SBATCH --account=<account>
# #SBATCH --reservation=<reservation>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ------ Environment -------------------------------------------------------
# Load modules / activate environment as needed for your cluster.
# On CLAIX-23 (RWTH Aachen) with the cuda-12.4 environment:
if [[ -f "../set_env_claix23_cuda12.4.sh" ]]; then
    source "../set_env_claix23_cuda12.4.sh"
fi

echo "================================================================"
echo "Job:        ${SLURM_JOB_ID:-local}"
echo "Node:       $(hostname)"
echo "Script dir: ${SCRIPT_DIR}"
echo "Python:     $(which python)"
echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:       $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"not available\")')"
echo "================================================================"

mkdir -p results

# ------ Step 1: Build test-MLP artifacts (if not present) ----------------
if [[ ! -f "artifacts/mmcp_test_mlp_m5_cpu.pt" || ! -f "artifacts/mmcp_test_mlp_m5_cuda.pt" ]]; then
    echo ""
    echo ">>> Building test MLP artifacts ..."
    python build_test_mlp_artifacts.py --device both
else
    echo ""
    echo ">>> Test MLP artifacts already present, skipping build."
    # Also build CUDA artifact if only CPU is present
    if [[ ! -f "artifacts/mmcp_test_mlp_m5_cuda.pt" ]]; then
        echo "    (CUDA artifacts missing, rebuilding ...)"
        python build_test_mlp_artifacts.py --device cuda
    fi
fi

# ------ Step 2: Static inspection only (CPU, all models) -----------------
echo ""
echo ">>> Step 2: Static inspection (CPU, --skip-timing) ..."
python inspect_models.py \
    --device cpu \
    --batch-sizes 1,8,32,128,512 \
    --skip-timing \
    ${SKIP_GIANT:+--no-giant} \
    ${MODELS:+--models "${MODELS}"} \
    --output-dir results

# ------ Step 3: Full GPU inspection (timing + activation memory) ----------
echo ""
echo ">>> Step 3: Full GPU inspection ..."

# Default batch sizes for GPU run: covers latency baseline through throughput regime.
# Matches the default in inspect_models.py; adjust BATCH_SIZES_GPU to override.
BATCH_SIZES_GPU="${BATCH_SIZES_GPU:-1,8,32,128,512}"

# Run all models except giant_mlp first (avoids OOM risk from large batch)
echo "   Running standard models (watercnn, perfect, transformer, mmcp models, test mlps) ..."
python inspect_models.py \
    --device cuda \
    --batch-sizes "${BATCH_SIZES_GPU}" \
    --models watercnn,perfect,transformer,mmcp_transformer_core,mmcp_transformer_5input,mmcp_test_mlp_m5,mmcp_test_mlp_m10 \
    --warmup 20 \
    --iters 100 \
    --timeout-s 120 \
    --output-dir results

# Run giant_mlp separately with a smaller batch cap (avoids OOM)
if [[ -z "${SKIP_GIANT:-}" ]]; then
    echo ""
    echo "   Running giant_mlp (batch sizes capped at 1,32 to avoid OOM) ..."
    python inspect_models.py \
        --device cuda \
        --batch-sizes 1,32 \
        --models giant_mlp \
        --warmup 5 \
        --iters 20 \
        --timeout-s 300 \
        --output-dir results
fi

echo ""
echo "================================================================"
echo "Inspection complete. Results in: ${SCRIPT_DIR}/results/"
ls -lh results/*.json results/*.md 2>/dev/null | tail -20
echo "================================================================"

# ------ Step 4: Roofline plots -------------------------------------------
# Find the most recent cuda inspection JSON and plot from it.
# If giant_mlp was run separately it produces its own JSON; we pass both so
# plot_roofline.py can merge them (model_id-keyed merge, later file wins for
# timing data).
echo ""
echo ">>> Step 4: Roofline plots ..."

CUDA_JSONS=( $(ls -t results/inspection_*_cuda.json 2>/dev/null | head -3) )
if [[ ${#CUDA_JSONS[@]} -gt 0 ]]; then
    python plot_roofline.py \
        "${CUDA_JSONS[@]}" \
        --output-dir results \
        --format pdf,png \
        --combined \
        --thesis
    echo "Roofline plots written to results/"
else
    echo "No cuda inspection JSON found; skipping roofline plots."
    echo "(Run the GPU inspection step first, then re-run plot_roofline.py manually.)"
fi

echo ""
echo "================================================================"
echo "All done. Results in: ${SCRIPT_DIR}/results/"
ls -lh results/*.pdf results/*.png 2>/dev/null | tail -10
echo "================================================================"
