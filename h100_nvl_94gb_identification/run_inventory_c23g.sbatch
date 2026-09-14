#!/bin/bash
#SBATCH -J h100_inventory
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH -p c23g
#SBATCH -A thes2181
#SBATCH -o h100_inventory_%j.out

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:?Submit from the directory containing the collector}"
DATA_DIR="${INVENTORY_REPORT_DIR:-${WORKDIR}/reports}"
mkdir -p "$DATA_DIR"

echo "Node: $(hostname)"
date

for gpu_index in 0 1 2 3; do
    report="${DATA_DIR}/h100_inventory_$(hostname -s)_gpu${gpu_index}_job${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ).txt"
    bash "$WORKDIR/collect_h100_inventory.sh" "$gpu_index" "$report"
done

echo "Inventory complete."
ls -l "$DATA_DIR"
