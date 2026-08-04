#!/bin/bash
#SBATCH --job-name=scorep-papi-hwctr-smoke
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --hwctr=papi
#SBATCH --output=scorep_papi_hwctr_smoke_%j.out
#SBATCH --error=scorep_papi_hwctr_smoke_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

export USE_SCOREP=1
source ./set_env_claix23_cuda12.4.sh

workdir="scorep_papi_hwctr_smoke_${SLURM_JOB_ID}"
mkdir -p "${workdir}"
cd "${workdir}"

cat > smoke.c <<'EOF'
#include <stdint.h>
#include <stdio.h>

int main(void) {
    volatile double x = 1.0;
    for (uint64_t i = 0; i < 100000000ULL; ++i) {
        x = x * 1.0000000001 + 0.0000000001;
    }
    printf("x=%f\n", x);
    return 0;
}
EOF

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Checking Score-P and PAPI availability"
which scorep-gcc
which papi_avail || true
which papi_component_avail || true
which papi_native_avail || true
scorep-info config-vars | grep -E 'SCOREP_METRIC_PAPI|SCOREP_ENABLE_PROFILING' || true
papi_component_avail || true
papi_avail || true

scorep-gcc smoke.c -O2 -o smoke

export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=false

for metric in "" "PAPI_TOT_INS" "PAPI_TOT_CYC" "PAPI_FP_OPS" "ix86arch::INSTRUCTION_RETIRED" "ix86arch::UNHALTED_CORE_CYCLES"; do
    safe_metric="${metric:-empty}"
    export SCOREP_EXPERIMENT_DIRECTORY="scorep-papi-hwctr-${safe_metric}"
    export SCOREP_METRIC_PAPI="${metric}"

    echo "----------------------------------------"
    echo "Running instrumented smoke program with SCOREP_METRIC_PAPI='${SCOREP_METRIC_PAPI}'"
    if ./smoke; then
        echo "Run succeeded for metric '${safe_metric}'"
    else
        echo "Run failed for metric '${safe_metric}'"
    fi

    if [[ -d "${SCOREP_EXPERIMENT_DIRECTORY}" ]]; then
        echo "Result directory contents for '${safe_metric}':"
        ls -la "${SCOREP_EXPERIMENT_DIRECTORY}"
        if [[ -f "${SCOREP_EXPERIMENT_DIRECTORY}/profile.cubex" ]]; then
            echo "Score-P profile summary for '${safe_metric}':"
            scorep-score "${SCOREP_EXPERIMENT_DIRECTORY}/profile.cubex" || true
        fi
    fi
done
