#!/usr/bin/env bash
set -euo pipefail

#PROVIDER="SMARTSIM" # Alternatives: "AIX", "PHYDLL" or "SMARTSIM"
PROVIDER="AIX"
export PROVIDER

USE_GPU=1
PYTHON_RUNTIME_ROOT="/home/thes2181/python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( USE_GPU == 1 )); then
    SMARTSIM_PYTHON="${SMARTSIM_PYTHON:-${PYTHON_RUNTIME_ROOT}/smartsim_cuda-12/bin/python}"
else
    SMARTSIM_PYTHON="${SMARTSIM_PYTHON:-${PYTHON_RUNTIME_ROOT}/smartsim_cpu/bin/python}"
fi

echo "A"


if (( USE_GPU == 1 )); then
	RUNTIME_DEVICE="smartsim_cuda-12"
	PY_ENV="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}"
	CONFIG_FILE="${SCRIPT_DIR}/config_smartsim_gpu.toml"
else
	RUNTIME_DEVICE="smartsim_cpu"
	PY_ENV="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}"
	CONFIG_FILE="${SCRIPT_DIR}/config_smartsim_cpu.toml"
fi

# Overwrite config file in case of AIX provider
if [[ "$PROVIDER" == "AIX" ]]; then
	CONFIG_FILE="${SCRIPT_DIR}/config_aix.toml"
elif [[ "$PROVIDER" == "SMARTSIM" ]]; then
	RUNTIME_EXTRA_LIB_DIR="${PY_ENV}/runtime_libs"
	if [[ -d "${RUNTIME_EXTRA_LIB_DIR}" ]]; then
		export LD_LIBRARY_PATH="${RUNTIME_EXTRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
		echo "Using runtime extra libs from ${RUNTIME_EXTRA_LIB_DIR}"
	else
		echo "Warning: runtime extra lib directory not found: ${RUNTIME_EXTRA_LIB_DIR}"
	fi
fi

echo "B"

# For SmartSim
if [[ $PROVIDER == "SMARTSIM" ]]; then
	ENDPOINT_FILE="${SCRIPT_DIR}/.ssdb_endpoint"
	DONE_FILE="${SCRIPT_DIR}/.solver_done"

	rm -f "${ENDPOINT_FILE}" "${DONE_FILE}"

	"${SMARTSIM_PYTHON}" "${SCRIPT_DIR}/driver.py" \
		--endpoint-file "${ENDPOINT_FILE}" \
		--done-file "${DONE_FILE}" &
	DRIVER_PID=$!

	cleanup() {
		if [[ -n "${DRIVER_PID:-}" ]] && kill -0 "${DRIVER_PID}" 2>/dev/null; then
			touch "${DONE_FILE}" || true
			wait "${DRIVER_PID}" || true
		fi
	}
	trap cleanup EXIT

	for _ in {1..120}; do
		if [[ -s "${ENDPOINT_FILE}" ]]; then
			break
		fi
		sleep 0.5
	done

	if [[ ! -s "${ENDPOINT_FILE}" ]]; then
		echo "Timed out waiting for SmartSim endpoint file: ${ENDPOINT_FILE}" >&2
		exit 1
	fi


	export SSDB
	SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"
	echo "Using SSDB=${SSDB}"

	cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
		-DSMARTSIM_PYTHON="${SMARTSIM_PYTHON}"
	cmake --build "${SCRIPT_DIR}/build" -j

	"${SCRIPT_DIR}/build/module_test_solver" "${CONFIG_FILE}"

	touch "${DONE_FILE}"
	wait "${DRIVER_PID}"
elif [[ $PROVIDER == "AIX" ]]; then

	echo "C"
	
	# srun --export=ALL --het-group=0 --mpi=pmi2 --preserve-env --cpus-per-task=1 /home/thes1961/MAIA/build_interface_aix_scorep_23b/bin/maia ./"$(basename $TOML_FILE)" : --export=ALL --het-group=1 --mpi=pmi2 --preserve-env --cpus-per-task=1 /home/thes1961/MAIA/build_interface_aix_scorep_23b/bin/maia ./"$(basename $TOML_FILE)"
	export CUDA_VISIBLE_DEVICES=3
	mpirun -n 1 "${SCRIPT_DIR}/build/module_test_solver" "${CONFIG_FILE}"
else
	echo "Unsupported provider: ${PROVIDER}" >&2
	exit 1
fi
