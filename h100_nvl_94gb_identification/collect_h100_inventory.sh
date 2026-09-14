#!/usr/bin/env bash
# Collect read-only NVIDIA, PCI, and CUDA inventory for one physical GPU.
set -euo pipefail

usage() {
    printf 'Usage: %s [physical-gpu-index] [report-path]\n' "${0##*/}" >&2
    printf 'Example: %s 3 h100_inventory.txt\n' "${0##*/}" >&2
}

gpu_index="${1:-3}"
if [[ $# -gt 2 || ! "$gpu_index" =~ ^[0-9]+$ ]]; then
    usage
    exit 2
fi

for command in nvidia-smi lspci python3; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command" >&2
        exit 1
    fi
done

gpu_uuid="$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
pci_bus_id="$(nvidia-smi -i "$gpu_index" --query-gpu=pci.bus_id --format=csv,noheader | tr -d '[:space:]')"
# NVIDIA uses an eight-digit domain; Linux sysfs uses four and lowercase hex.
IFS=: read -r pci_domain pci_bus pci_slot <<< "$pci_bus_id"
printf -v pci_bdf '%04x:%s:%s' "$((16#$pci_domain))" "${pci_bus,,}" "${pci_slot,,}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
report_path="${2:-h100_inventory_$(hostname -s)_gpu${gpu_index}_${timestamp}.txt}"

mkdir -p "$(dirname "$report_path")"
set -o noclobber

# Run an optional command without aborting the report on failure.
run_optional() {
    local description="$1"
    shift
    printf '## %s\n\n' "$description"
    if "$@" 2>&1; then
        :
    else
        local status=$?
        printf '[command failed with status %d; output above may be incomplete]\n' "$status"
    fi
    printf '\n'
}

{
    printf '# NVIDIA H100 hardware inventory\n\n'
    printf 'Collected (UTC): %s\n' "$(date -u --iso-8601=seconds)"
    printf 'Hostname: %s\n' "$(hostname -f)"
    printf 'Kernel: %s\n' "$(uname -srmo)"
    printf 'SLURM_JOB_ID: %s\n' "${SLURM_JOB_ID:-not-set}"
    printf 'SLURM_JOB_NODELIST: %s\n' "${SLURM_JOB_NODELIST:-not-set}"
    printf 'SLURM_JOB_PARTITION: %s\n' "${SLURM_JOB_PARTITION:-not-set}"
    printf 'SLURM_JOB_ACCOUNT: %s\n' "${SLURM_JOB_ACCOUNT:-not-set}"
    printf 'Target physical GPU index: %s\n' "$gpu_index"
    printf 'Target GPU UUID: %s\n' "$gpu_uuid"
    printf 'Target PCI bus ID: %s\n\n' "$pci_bus_id"

    printf '## Driver Summary\n\n'
    nvidia-smi

    printf '\n## All Physical GPUs\n\n'
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,pci.device_id,pci.sub_device_id,vbios_version,memory.total,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory,pstate,mig.mode.current --format=csv

    printf '\n## MIG Inventory\n\n'
    nvidia-smi -L

    printf '\n## Target Full Query (product, board, PCIe, power)\n\n'
    nvidia-smi -q -i "$gpu_index"

    printf '\n## GPU Topology\n\n'
    nvidia-smi topo -m

    printf '\n## NVLink Link State\n\n'
    run_optional "NVLink summary" nvidia-smi nvlink -s -i "$gpu_index"
    run_optional "NVLink capabilities" nvidia-smi nvlink -c -i "$gpu_index"
    run_optional "NVLink remote PCI endpoints" nvidia-smi nvlink -p -i "$gpu_index"

    printf '\n## Target PCI Identification\n\n'
    lspci -nn -s "$pci_bdf"

    printf '\n## Target PCI Link Capabilities (lspci verbose)\n\n'
    run_optional "lspci -vv (LnkCap/LnkSta)" lspci -vv -s "$pci_bdf"

    printf '\n## Target PCIe Sysfs Link Attributes\n\n'
    sysfs_pci="/sys/bus/pci/devices/${pci_bdf}"
    printf 'Sysfs device: %s\n' "$sysfs_pci"
    for attribute in current_link_speed max_link_speed current_link_width max_link_width numa_node; do
        if [[ -r "${sysfs_pci}/${attribute}" ]]; then
            IFS= read -r value < "${sysfs_pci}/${attribute}"
            printf '%s: %s\n' "$attribute" "$value"
        else
            printf '%s: not readable\n' "$attribute"
        fi
    done

    printf '\n## Target Supported Clocks\n\n'
    nvidia-smi -q -i "$gpu_index" -d SUPPORTED_CLOCKS

    printf '\n## Host NUMA Topology\n\n'
    run_optional "CPU/socket/NUMA summary" lscpu
    run_optional "CPU/socket/NUMA mapping" lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
    if command -v numactl >/dev/null 2>&1; then
        numactl --hardware
    else
        printf 'numactl not available\n'
    fi

    printf '\n## CUDA Device Attributes\n\n'
    CUDA_VISIBLE_DEVICES="$gpu_uuid" python3 - <<'PY'
import ctypes as c

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

cuda = c.CDLL("libcuda.so.1")
cuda.cuInit.argtypes = [c.c_uint]
cuda.cuDeviceGet.argtypes = [c.POINTER(c.c_int), c.c_int]
cuda.cuDeviceGetAttribute.argtypes = [c.POINTER(c.c_int), c.c_int, c.c_int]
cuda.cuDeviceTotalMem_v2.argtypes = [c.POINTER(c.c_size_t), c.c_int]

def check(status, operation):
    if status != CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")

def attribute(device, identifier):
    value = c.c_int()
    check(cuda.cuDeviceGetAttribute(c.byref(value), identifier, device), "cuDeviceGetAttribute")
    return value.value

device = c.c_int()
check(cuda.cuInit(0), "cuInit")
check(cuda.cuDeviceGet(c.byref(device), 0), "cuDeviceGet")

sm_count = attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
bus_width = attribute(device, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
cc_major = attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
cc_minor = attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
memory_bytes = c.c_size_t()
check(cuda.cuDeviceTotalMem_v2(c.byref(memory_bytes), device), "cuDeviceTotalMem_v2")

print(f"SM count: {sm_count}")
print(f"Global memory bus width: {bus_width} bits")
print(f"Compute capability: {cc_major}.{cc_minor}")
print(f"CUDA total memory: {memory_bytes.value} bytes ({memory_bytes.value / 2**30:.6f} GiB)")
if (cc_major, cc_minor) == (9, 0):
    print(f"Derived Hopper CUDA cores: {sm_count * 128}")
    print(f"Derived Hopper Tensor Cores: {sm_count * 4}")
    print(f"Derived Hopper TMUs: {sm_count * 4}")
PY
} >"$report_path"

printf 'Wrote inventory report: %s\n' "$report_path"
