"""Artifact-driven inspection: storage, timing, VRAM scaling, and link calibration.

The loaded TorchScript artifact is authoritative. Storage and I/O contracts come
from the loaded module and an actual forward call; latency and throughput come
from CUDA events; arithmetic and memory traffic are measured separately with
Nsight Compute counters.
"""

from __future__ import annotations

import collections
import hashlib
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from catalog import flatten_tensors, make_inputs, tensor_bytes

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visible_gpu_info(device_index: int) -> dict[str, Any]:
    """Best-effort clocks of the CUDA-visible GPU using nvidia-smi."""
    import os

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,clocks.max.sm,clocks.max.mem", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        entries = [line.split(", ") for line in output.strip().splitlines() if line.strip()]
    except Exception as exc:
        return {"clocks_unavailable": str(exc)}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    target = None
    if visible:
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if len(tokens) == 1:
            token = tokens[0]
            if token.startswith("GPU-"):
                target = next((entry for entry in entries if entry[1] == token), None)
            elif token.isdigit():
                target = next((entry for entry in entries if entry[0] == token), None)
    if target is None:
        target = next((entry for entry in entries if entry[0] == str(device_index)), None)
    if target is None:
        return {}
    return {"max_sm_clock_mhz": int(target[2]), "max_mem_clock_mhz": int(target[3])}


def gpu_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if device.type != "cuda" or not torch.cuda.is_available():
        return info
    properties = torch.cuda.get_device_properties(device)
    info = {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
    }
    info.update(_visible_gpu_info(device.index or 0))
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30, check=True).stdout
        info["nvidia_smi_devices"] = [line.strip() for line in listing.splitlines() if line.strip()]
    except Exception as exc:
        info["nvidia_smi_devices_unavailable"] = str(exc)
    return info


def configure_precision(policy: str) -> None:
    if not torch.cuda.is_available():
        return
    enabled = policy == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def storage_summary(model: torch.nn.Module) -> dict[str, Any]:
    seen: set[tuple[int, int]] = set()
    groups: dict[str, dict[str, int]] = {"parameters": collections.Counter(), "buffers": collections.Counter()}
    bytes_by_group = {"parameters": 0, "buffers": 0}
    for group, tensors in (("parameters", list(model.parameters())), ("buffers", list(model.buffers()))):
        for tensor in tensors:
            groups[group][str(tensor.dtype)] += tensor.numel()
            key = (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            if key not in seen:
                seen.add(key)
                bytes_by_group[group] += tensor.numel() * tensor.element_size()
    return {
        "parameter_elements": sum(groups["parameters"].values()),
        "buffer_elements": sum(groups["buffers"].values()),
        "parameter_bytes": bytes_by_group["parameters"],
        "buffer_bytes": bytes_by_group["buffers"],
        "resident_tensor_bytes": bytes_by_group["parameters"] + bytes_by_group["buffers"],
        "dtype_elements": {group: dict(values) for group, values in groups.items()},
        "deduplicates_aliased_storage": True,
    }


def measure_workspace(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    if inputs[0].device.type != "cuda":
        return {"method": "not_available_without_cuda"}
    torch.cuda.synchronize(inputs[0].device)
    torch.cuda.reset_peak_memory_stats(inputs[0].device)
    baseline = torch.cuda.memory_allocated(inputs[0].device)
    with torch.inference_mode():
        output = model(*inputs)
    torch.cuda.synchronize(inputs[0].device)
    peak = torch.cuda.max_memory_allocated(inputs[0].device)
    output_bytes = tensor_bytes(flatten_tensors(output))
    return {
        "method": "cuda_max_memory_allocated_delta",
        "input_bytes": tensor_bytes(list(inputs)),
        "output_bytes": output_bytes,
        "peak_incremental_bytes": max(0, peak - baseline),
        "workspace_and_intermediate_upper_bound_bytes": max(0, peak - baseline - output_bytes),
        "note": "Capacity measurement only; physical HBM traffic is collected separately with Nsight Compute.",
    }


def med(values: list[float]) -> float:
    return float(statistics.median(values))


def benchmark_cuda(model: torch.nn.Module, spec: dict[str, Any], batch_size: int, device: torch.device, warmup: int, iterations: int) -> dict[str, Any]:
    host_inputs = make_inputs(spec, batch_size, torch.device("cpu"), pinned=True)
    device_inputs = tuple(torch.empty_like(tensor, device=device) for tensor in host_inputs)
    with torch.inference_mode():
        for _ in range(warmup):
            for destination, source in zip(device_inputs, host_inputs):
                destination.copy_(source, non_blocking=True)
            output = model(*device_inputs)
        torch.cuda.synchronize(device)
    host_outputs = [torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True) for tensor in flatten_tensors(output)]
    events = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
    h2d, forward, d2h, step, resident = [], [], [], [], []
    for _ in range(iterations):
        events[6].record(); events[0].record()
        for destination, source in zip(device_inputs, host_inputs):
            destination.copy_(source, non_blocking=True)
        events[1].record(); events[2].record()
        with torch.inference_mode():
            output = model(*device_inputs)
        events[3].record(); events[4].record()
        for destination, source in zip(host_outputs, flatten_tensors(output)):
            destination.copy_(source, non_blocking=True)
        events[5].record(); events[7].record()
        torch.cuda.synchronize(device)
        h2d.append(events[0].elapsed_time(events[1]))
        forward.append(events[2].elapsed_time(events[3]))
        d2h.append(events[4].elapsed_time(events[5]))
        step.append(events[6].elapsed_time(events[7]))
    for _ in range(iterations):
        events[2].record()
        with torch.inference_mode():
            model(*device_inputs)
        events[3].record()
        torch.cuda.synchronize(device)
        resident.append(events[2].elapsed_time(events[3]))
    resident_ms = med(resident)
    step_ms = med(step)
    return {
        "batch_size": batch_size,
        "input_bytes": tensor_bytes(list(host_inputs)),
        "output_bytes": tensor_bytes(host_outputs),
        "median_h2d_ms": med(h2d),
        "median_forward_ms": med(forward),
        "median_d2h_ms": med(d2h),
        "median_step_ms": step_ms,
        "median_transfer_ms": med([a + b for a, b in zip(h2d, d2h)]),
        "median_resident_forward_ms": resident_ms,
        "resident_samples_per_s": batch_size / (resident_ms / 1000) if resident_ms > 0 else None,
        "serial_samples_per_s": batch_size / (step_ms / 1000) if step_ms > 0 else None,
        "iterations": iterations,
    }


def benchmark_cpu(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...], iterations: int) -> dict[str, Any]:
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.inference_mode():
            model(*inputs)
        elapsed.append((time.perf_counter() - start) * 1000)
    return {"batch_size": inputs[0].shape[0], "median_forward_ms": med(elapsed), "iterations": iterations}


def calibrate_link(device: torch.device, bytes_per_direction: int, iterations: int) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    elements = bytes_per_direction // 4
    host = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    gpu = torch.empty_like(host, device=device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    h2d, d2h = [], []
    for _ in range(5):
        gpu.copy_(host, non_blocking=True)
    torch.cuda.synchronize(device)
    for _ in range(iterations):
        start.record(); gpu.copy_(host, non_blocking=True); end.record(); torch.cuda.synchronize(device)
        h2d.append(start.elapsed_time(end))
        start.record(); host.copy_(gpu, non_blocking=True); end.record(); torch.cuda.synchronize(device)
        d2h.append(start.elapsed_time(end))
    return {
        "bytes_per_direction": bytes_per_direction,
        "median_h2d_gbps": bytes_per_direction / (med(h2d) / 1000) / 1e9,
        "median_d2h_gbps": bytes_per_direction / (med(d2h) / 1000) / 1e9,
        "method": "pinned_host_single_stream_cuda_event",
    }


def runtime_graph_audit() -> dict[str, Any]:
    try:
        graph = str(torch.jit.last_executed_optimized_graph())
        return {"optimized_graph_sha256": hashlib.sha256(graph.encode()).hexdigest(), "optimized_graph_lines": graph.count("\n") + 1}
    except Exception as exc:
        return {"optimized_graph_unavailable": str(exc)}


def measure_vram(model: torch.nn.Module, spec: dict[str, Any], batch: int, device: torch.device) -> dict[str, int]:
    """Run one forward at ``batch`` and report peak VRAM usage.

    The peak is measured with the model already resident, so it includes the
    weights and any other persistent state. Raises if the forward cannot fit.
    """
    inputs = make_inputs(spec, batch, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = model(*inputs)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del inputs, output
    torch.cuda.empty_cache()
    return {"peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}


def linear_memory_fit(points: list[dict[str, Any]], total_memory_bytes: int) -> dict[str, Any]:
    """Least-squares fit of peak VRAM over batch size and the derived limits."""
    import numpy as np

    xs = [point["batch"] for point in points]
    ys = [point["peak_allocated_bytes"] for point in points]
    if len(xs) >= 2:
        slope, intercept = (float(value) for value in np.polyfit(xs, ys, 1))
        predicted = [intercept + slope * x for x in xs]
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
        ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    else:
        slope, intercept, r2 = 0.0, float(ys[0]), None
    result = {
        "intercept_bytes": intercept,
        "slope_bytes_per_sample": slope,
        "r_squared": r2,
        "total_memory_bytes": total_memory_bytes,
    }
    if slope > 0:
        result["predicted_max_safe_batch_90pct"] = int((0.9 * total_memory_bytes - intercept) / slope)
        result["predicted_100pct_batch"] = int((total_memory_bytes - intercept) / slope)
    return result


def discover_batches(model: torch.nn.Module, spec: dict[str, Any], device: torch.device,
                     memory_fraction: float = 0.9, batch_step: int = 10, max_batch: int | None = None,
                     manual: list[int] | None = None, logger=None) -> dict[str, Any]:
    """Adaptively grow the batch ladder until predicted VRAM crosses the gate.

    With ``manual`` the listed ladder is measured directly (OOM-guarded) and no
    prediction is used. Otherwise the ladder starts at 1 and the next candidate
    is the current largest times ``batch_step``; a two-point line through B=1
    and the current largest predicts the candidate, and growth stops once the
    prediction exceeds ``memory_fraction`` of total device memory.
    """
    total_memory = torch.cuda.get_device_properties(device).total_memory
    gate = int(total_memory * memory_fraction)
    points: list[dict[str, Any]] = []
    stop_reason = None

    def attempt(batch: int) -> bool:
        try:
            record = measure_vram(model, spec, batch, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            nonlocal stop_reason
            stop_reason = f"out of memory at B={batch}: {type(exc).__name__}"
            torch.cuda.empty_cache()
            return False
        points.append({"batch": batch, **record})
        if logger:
            logger.info(f"    VRAM B={batch}: {record['peak_allocated_bytes'] / 2**30:.2f} GiB allocated, "
                        f"{record['peak_reserved_bytes'] / 2**30:.2f} GiB reserved")
        return True

    if manual:
        for batch in sorted(set(manual)):
            if max_batch is not None and batch > max_batch:
                stop_reason = f"manual batch {batch} exceeds --max-batch {max_batch}"
                break
            if not attempt(batch):
                break
        if stop_reason is None:
            stop_reason = "manual ladder exhausted"
    else:
        if not attempt(1):
            raise RuntimeError("could not fit B=1; refusing to continue")
        while True:
            current = points[-1]["batch"]
            candidate = current * batch_step
            if max_batch is not None and candidate > max_batch:
                stop_reason = f"next candidate B={candidate} exceeds --max-batch {max_batch}"
                break
            if len(points) >= 2:
                first = points[0]
                slope = (points[-1]["peak_allocated_bytes"] - first["peak_allocated_bytes"]) / (current - first["batch"])
                intercept = first["peak_allocated_bytes"] - slope * first["batch"]
                predicted = intercept + slope * candidate
                if predicted > gate:
                    stop_reason = (f"predicted B={candidate} needs {predicted / 2**30:.1f} GiB > "
                                   f"{memory_fraction:.0%} of {total_memory / 2**30:.1f} GiB")
                    break
            if not attempt(candidate):
                break
    return {
        "ladder": [point["batch"] for point in points],
        "memory_scaling": points,
        "stop_reason": stop_reason,
        "memory_fraction": memory_fraction,
        "batch_step": batch_step,
        "fit": linear_memory_fit(points, total_memory),
    }


def run_ncu_forward(model: torch.nn.Module, spec: dict[str, Any], batch_size: int, device: torch.device) -> None:
    inputs = make_inputs(spec, batch_size, device)
    with torch.inference_mode():
        for _ in range(5):
            model(*inputs)
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_push("model_forward")
    with torch.inference_mode():
        model(*inputs)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)


def run_nsys_step(model: torch.nn.Module, spec: dict[str, Any], batch_size: int, device: torch.device, repeats: int) -> None:
    host_inputs = make_inputs(spec, batch_size, torch.device("cpu"), pinned=True)
    device_inputs = tuple(torch.empty_like(tensor, device=device) for tensor in host_inputs)
    with torch.inference_mode():
        for _ in range(3):
            model(*device_inputs)
        output = model(*device_inputs)
    torch.cuda.synchronize(device)
    host_outputs = [torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True) for tensor in flatten_tensors(output)]
    h2d_bytes = tensor_bytes(list(host_inputs))
    d2h_bytes = tensor_bytes(host_outputs)
    torch.cuda.nvtx.range_push("nsys_audit")
    with torch.inference_mode():
        for _ in range(repeats):
            torch.cuda.nvtx.range_push("audit_h2d")
            for destination, source in zip(device_inputs, host_inputs):
                destination.copy_(source, non_blocking=True)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("audit_forward")
            output = model(*device_inputs)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("audit_d2h")
            for destination, source in zip(host_outputs, flatten_tensors(output)):
                destination.copy_(source, non_blocking=True)
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_pop()
    print(f"NSYS_TARGET expected per-repeat h2d_bytes={h2d_bytes} d2h_bytes={d2h_bytes} repeats={repeats}", flush=True)
