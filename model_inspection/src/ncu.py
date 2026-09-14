"""Nsight Compute collection: launch NCU per model/precision/batch and parse CSVs.

The CUDA 12.4-bundled NCU (2024.1) is incompatible with driver 580.x; the
pipeline defaults to the CUDA 12.8 NCU (2025.1) binary. CSV numbers may carry
locale-dependent thousands separators and scaled units, so parsing is
structural and tolerant.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

DRAM_METRICS = ["dram__bytes.sum", "dram__bytes_read.sum", "dram__bytes_write.sum"]
L2_METRICS = ["lts__t_bytes.sum", "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum"]
ARITHMETIC_METRICS = [
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
]
TENSOR_METRICS = [
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp16.sum",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp8.sum",
    "sm__ops_path_tensor_src_int8.sum",
    "sm__ops_path_tensor_src_fp64.sum",
]
DEFAULT_METRICS = DRAM_METRICS + L2_METRICS + ARITHMETIC_METRICS + TENSOR_METRICS
MISSING = "-"

# NCU reports cache-sector counters in sectors; 32 bytes per sector on H100.
_BYTE_UNITS = {"byte": 1, "Kbyte": 1024, "Mbyte": 1024**2, "Gbyte": 1024**3,
               "sector": 32, "sectors": 32}


def parse_number(token: str) -> float | None:
    text = token.strip()
    if not text or text in {"-", "n/a"}:
        return None
    cleaned = text.replace(",", "")
    parts = cleaned.split(".")
    if len(parts) > 2 and parts[0] != "" and all(part.isdigit() and len(part) == 3 for part in parts[1:]):
        cleaned = "".join(parts)
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_metrics_csv(path: Path, metrics: list[str]) -> dict | None:
    """Sum each requested metric over all kernel rows in one NCU CSV."""
    totals = {name: MISSING for name in metrics}
    kernels = 0
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or len(row) < 2:
                continue
            matched = None
            for name in metrics:
                if name in row:
                    matched = name
                    break
            if matched is None:
                continue
            value = None
            index = row.index(matched)
            if index + 2 < len(row):
                value = parse_number(row[index + 2])
                if value is not None:
                    value *= _BYTE_UNITS.get(row[index + 1], 1)
            if value is None:
                for token in reversed(row):
                    value = parse_number(token)
                    if value is not None:
                        break
            if value is None:
                continue
            if totals[matched] == MISSING:
                totals[matched] = 0.0
            totals[matched] += value
            if matched == metrics[0]:
                kernels += 1
    if kernels == 0 and all(totals[name] == MISSING for name in metrics):
        return None
    result = {"kernel_count": kernels, "counters": totals}
    dram = totals.get("dram__bytes.sum", MISSING)
    if dram == MISSING or dram == 0:
        total = 0.0
        for name in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
            value = totals.get(name, MISSING)
            if value != MISSING:
                total += value
        result["dram_bytes_total"] = total
    else:
        result["dram_bytes_total"] = dram
    result["dram_bytes_read"] = totals.get("dram__bytes_read.sum", 0.0) if totals.get("dram__bytes_read.sum", MISSING) != MISSING else 0.0
    result["dram_bytes_write"] = totals.get("dram__bytes_write.sum", 0.0) if totals.get("dram__bytes_write.sum", MISSING) != MISSING else 0.0
    for key, metric in (("l2_bytes_total", "lts__t_bytes.sum"),
                        ("l2_bytes_read", "lts__t_sectors_op_read.sum"),
                        ("l2_bytes_write", "lts__t_sectors_op_write.sum")):
        value = totals.get(metric, MISSING)
        result[key] = 0.0 if value == MISSING else value
    return result


def entry_ok(entry: dict) -> bool:
    return entry.get("kernel_count", 0) > 0


def ncu_version(ncu_bin: Path) -> str:
    try:
        version = subprocess.run([str(ncu_bin), "--version"], capture_output=True, text=True, timeout=120).stdout
        return next((line for line in version.splitlines() if "Version" in line), version.strip().splitlines()[-1] if version else "?")
    except Exception:
        return "unknown"


def collect(catalog_path: Path, models: list[str], precisions: list[str], batches_by_model: dict[str, list[int]],
            ncu_bin: Path, metrics: list[str], run_dir: Path, timeout: int, logger,
            cache_control: str = "none", replay_mode: str = "range", clock_control: str = "none") -> list[dict]:
    script_dir = Path(__file__).resolve().parent
    env = dict(os.environ, LC_ALL="C")
    entries: list[dict] = []
    for precision in precisions:
        for model in models:
            for batch in batches_by_model.get(model, []):
                csv_path = run_dir / "raw" / f"ncu_{model}_{precision}_b{batch}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(ncu_bin), "--target-processes", "all", "--nvtx", "--nvtx-include", "model_forward/",
                    "--cache-control", cache_control, "--replay-mode", replay_mode, "--clock-control", clock_control,
                    "--csv", "--metrics", ",".join(metrics),
                    sys.executable, str(script_dir / "forward_target.py"),
                    "--catalog", str(catalog_path), "--model", model, "--batch", str(batch),
                    "--device", "cuda:0", "--precision", precision, "--mode", "ncu",
                ]
                logger.info("$ " + " ".join(command))
                with csv_path.open("w") as handle:
                    try:
                        code = subprocess.run(command, env=env, timeout=timeout, stdout=handle, stderr=subprocess.STDOUT).returncode
                    except subprocess.TimeoutExpired:
                        logger.error(f"  NCU timeout after {timeout}s for {model}/{precision}/B={batch}")
                        code = 124
                parsed = parse_metrics_csv(csv_path, metrics) if csv_path.exists() else None
                entry = {"model": model, "precision": precision, "batch": batch, "csv": str(csv_path.resolve()), "exit_code": code,
                         "cache_mode": cache_control, "replay_mode": replay_mode, "clock_control": clock_control}
                if parsed and entry_ok(parsed):
                    entry.update(parsed)
                    dram = parsed["dram_bytes_total"]
                    logger.info(f"  NCU {model}/{precision}/B={batch}: {dram:,.0f} B DRAM, "
                                f"{parsed.get('l2_bytes_total', 0):,.0f} B L2 over {parsed['kernel_count']} kernels "
                                f"({cache_control} cache, {replay_mode} replay)")
                else:
                    entry.update({"kernel_count": 0, "dram_bytes_total": 0, "dram_bytes_read": 0, "dram_bytes_write": 0, "counters": {name: MISSING for name in metrics}})
                    logger.error(f"  NCU {model}/{precision}/B={batch}: no usable counters (exit {code})")
                entries.append(entry)
    return entries
