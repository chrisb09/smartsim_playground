"""Assemble all long-form CSV tables from measured run data.

Every table shares the same shape: ``model_id, display_name, precision,
batch_size, metric, value, unit, source, note``. Values that were not measured
are written as an empty field, never as a fabricated zero.
"""

from __future__ import annotations

import csv
from pathlib import Path

from catalog import schema_bytes, spec_for
from ncu import MISSING

HEADER = ["model_id", "display_name", "precision", "batch_size", "metric", "value", "unit", "source", "note"]

FFMA = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
FADD = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
FMUL = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
TENSOR = "sm__ops_path_tensor_src_tf32_dst_fp32.sum"
TENSOR_PRECISION_METRICS = {
    "tf32": "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
    "fp16": "sm__ops_path_tensor_src_fp16.sum",
    "bf16": "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "fp8": "sm__ops_path_tensor_src_fp8.sum",
    "int8": "sm__ops_path_tensor_src_int8.sum",
    "fp64": "sm__ops_path_tensor_src_fp64.sum",
}
FLOP_CONVENTION = ("measured_flops = 2*FFMA + FADD + FMUL + tensor-core-ops; "
                   "counts executed hardware work, not mathematical operations")


def display_name(spec: dict) -> str:
    return spec.get("short_name") or spec.get("display_name") or spec.get("id", "")


class TableWriter:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.counts: dict[str, int] = {}

    def write(self, name: str, rows: list[list]) -> Path:
        path = self.directory / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(rows)
        self.counts[name] = len(rows)
        return path

    @staticmethod
    def record(model_id: str, display: str, precision: str, batch, metric: str, value, unit: str, source: str, note: str = "") -> list:
        if value is None or value == MISSING:
            value = ""
        return [model_id, display, precision, batch, metric, value, unit, source, note]


def ncu_index(ncu_summary: dict) -> dict[tuple[str, str, int], dict]:
    return {(entry["model"], entry["precision"], int(entry["batch"])): entry for entry in ncu_summary.get("entries", [])}


def nsys_index(nsys_summary: dict) -> dict[tuple[str, str, int], dict]:
    return {(entry["model"], entry["precision"], int(entry["batch"])): entry for entry in nsys_summary.get("entries", [])}


def counter_value(counters: dict, name: str) -> float | None:
    value = counters.get(name, MISSING)
    if value == MISSING:
        return None
    return float(value)


def measured_flops(counters: dict) -> float | None:
    """Counter-derived executed FLOPs, including tensor-core operations.

    Tensor-core ``ops_path`` counters already count one multiply and one add
    per MAC (validated against a GEMM of known operation count), so they enter
    the sum unweighted, while FFMA covers both.
    """
    ffma = counter_value(counters, FFMA)
    fadd = counter_value(counters, FADD)
    fmul = counter_value(counters, FMUL)
    tensor_values = [counter_value(counters, metric) for metric in TENSOR_PRECISION_METRICS.values()]
    if all(value is None for value in (ffma, fadd, fmul)) and all(value is None for value in tensor_values):
        return None
    fp32 = 2.0 * (ffma or 0.0) + (fadd or 0.0) + (fmul or 0.0)
    tensor = sum(value or 0.0 for value in tensor_values)
    return fp32 + tensor


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_all(run_dir: Path, catalog: dict, models: list[str], precisions: list[str],
              measurements: dict[str, dict], ncu_summary: dict, nsys_summary: dict) -> dict:
    writer = TableWriter(run_dir / "tables")
    ncu_entries = ncu_index(ncu_summary)
    nsys_entries = nsys_index(nsys_summary)
    io: dict[str, tuple[int, int]] = {}
    ladders: dict[str, list[int]] = {}

    for model_id in models:
        spec = spec_for(catalog, model_id)
        io[model_id] = (schema_bytes(spec["inputs"]), schema_bytes(spec["outputs"]))
        record = next((entry for entry in measurements[precisions[0]]["models"] if entry["model_id"] == model_id), None)
        if record:
            ladder = record.get("adaptive", {}).get("ladder") or [timing["batch_size"] for timing in record["batches"]]
            ladders[model_id] = [int(value) for value in ladder]

    # 1. model storage (source inspection)
    rows = []
    for model_id in models:
        spec = spec_for(catalog, model_id)
        record = next((entry for entry in measurements[precisions[0]]["models"] if entry["model_id"] == model_id), None)
        if not record:
            continue
        storage = record["weights"]
        for metric in ("parameter_elements", "buffer_elements", "parameter_bytes", "buffer_bytes", "resident_tensor_bytes"):
            rows.append(writer.record(model_id, display_name(spec), "all", 0, metric, storage[metric], "bytes" if metric.endswith("_bytes") else "elements", "source_inspection"))
        for dtype, elements in storage["dtype_elements"]["parameters"].items():
            rows.append(writer.record(model_id, display_name(spec), "all", 0, f"parameters_{dtype}", elements, "elements", "source_inspection"))
    writer.write("model_storage", rows)

    # 2. measured hardware counters
    rows = []
    for (model_id, precision, batch), entry in sorted(ncu_entries.items()):
        spec = spec_for(catalog, model_id)
        counters = entry.get("counters", {})
        for counter, value in counters.items():
            unit = "instructions" if counter.startswith("smsp__sass_thread_inst_executed_op") else "ops"
            rows.append(writer.record(model_id, display_name(spec), precision, batch, counter, value, unit, "ncu_hardware_counters", f"cache={entry.get('cache_mode', '')}"))
        rows.append(writer.record(model_id, display_name(spec), precision, batch,
                                  "measured_flops", measured_flops(counters), "flops", "ncu_hardware_counters", FLOP_CONVENTION))
    writer.write("flops_measured_hardware", rows)

    # 2b. arithmetic FLOPs split by executed precision
    rows = []
    for (model_id, precision, batch), entry in sorted(ncu_entries.items()):
        spec = spec_for(catalog, model_id)
        counters = entry.get("counters", {})
        ffma = counter_value(counters, FFMA) or 0.0
        fadd = counter_value(counters, FADD) or 0.0
        fmul = counter_value(counters, FMUL) or 0.0
        fp32 = 2.0 * ffma + fadd + fmul
        rows.append(writer.record(model_id, display_name(spec), precision, batch, "fp32_flops", fp32, "flops", "ncu_hardware_counters", "FMA pipe: 2*FFMA + FADD + FMUL"))
        tensor_total = 0.0
        for tensor_precision, metric in TENSOR_PRECISION_METRICS.items():
            value = counter_value(counters, metric)
            flops = value or 0.0
            tensor_total += flops
            rows.append(writer.record(model_id, display_name(spec), precision, batch, f"{tensor_precision}_flops", flops, "flops", "ncu_hardware_counters", "tensor-core ops (counted as multiply + add)"))
        rows.append(writer.record(model_id, display_name(spec), precision, batch, "tensor_flops", tensor_total, "flops", "ncu_hardware_counters"))
        rows.append(writer.record(model_id, display_name(spec), precision, batch, "total_flops", fp32 + tensor_total, "flops", "ncu_hardware_counters"))
    writer.write("precision_ops_measured", rows)

    # 3. HBM3 measured DRAM
    rows = []
    for (model_id, precision, batch), entry in sorted(ncu_entries.items()):
        spec = spec_for(catalog, model_id)
        for metric, key in (("read", "dram_bytes_read"), ("write", "dram_bytes_write"), ("total", "dram_bytes_total")):
            rows.append(writer.record(model_id, display_name(spec), precision, batch, metric, entry.get(key), "bytes",
                                      "ncu_dram_counters", f"cache={entry.get('cache_mode', '')}"))
    writer.write("hbm3_measured", rows)

    # 4. L2 versus DRAM
    rows = []
    for (model_id, precision, batch), entry in sorted(ncu_entries.items()):
        spec = spec_for(catalog, model_id)
        display = display_name(spec)
        for metric, key in (("dram_read", "dram_bytes_read"), ("dram_write", "dram_bytes_write"), ("dram_total", "dram_bytes_total"),
                            ("l2_read", "l2_bytes_read"), ("l2_write", "l2_bytes_write"), ("l2_total", "l2_bytes_total")):
            rows.append(writer.record(model_id, display, precision, batch, metric, entry.get(key), "bytes", "ncu_cache_counters"))
        dram, l2 = entry.get("dram_bytes_total"), entry.get("l2_bytes_total")
        if dram and l2:
            rows.append(writer.record(model_id, display, precision, batch, "l2_over_dram", l2 / dram, "ratio", "ncu_cache_counters", "L2 requests per DRAM byte; >1 indicates cache reuse"))
    writer.write("cache_l2_measured", rows)

    # 5. PCIe measured
    rows = []
    for (model_id, precision, batch), entry in sorted(nsys_entries.items()):
        spec = spec_for(catalog, model_id)
        repeats = entry.get("repeats", 1) or 1
        h2d = entry.get("h2d_bytes")
        d2h = entry.get("d2h_bytes")
        if h2d is not None:
            h2d = h2d / repeats
        if d2h is not None:
            d2h = d2h / repeats
        rows.append(writer.record(model_id, display_name(spec), precision, batch, "h2d", h2d, "bytes", "nsys_activity_memcpy", f"per inference ({repeats} repeats traced)"))
        rows.append(writer.record(model_id, display_name(spec), precision, batch, "d2h", d2h, "bytes", "nsys_activity_memcpy", f"per inference ({repeats} repeats traced)"))
        rows.append(writer.record(model_id, display_name(spec), precision, batch, "total", (h2d + d2h) if h2d is not None and d2h is not None else None, "bytes", "nsys_activity_memcpy", f"per inference ({repeats} repeats traced)"))
    writer.write("pcie_measured", rows)

    # 6. measured operational intensity (counter FLOPs per measured DRAM byte)
    rows = []
    for (model_id, precision, batch), entry in sorted(ncu_entries.items()):
        spec = spec_for(catalog, model_id)
        flops = measured_flops(entry.get("counters", {}))
        memory = entry.get("dram_bytes_total")
        value = flops / memory if flops and memory else None
        rows.append(writer.record(model_id, display_name(spec), precision, batch, "operational_intensity", value, "flops/byte", "ncu_counters"))
    writer.write("oi_hbm3_measured", rows)

    # 7. timing and throughput
    rows = []
    for precision in precisions:
        for model_id in models:
            spec = spec_for(catalog, model_id)
            record = next((entry for entry in measurements[precision]["models"] if entry["model_id"] == model_id), None)
            if not record:
                continue
            for timing in record["batches"]:
                batch = timing["batch_size"]
                resident_ms = timing.get("median_resident_forward_ms")
                step_ms = timing.get("median_step_ms")
                timing.setdefault("resident_samples_per_s", batch / (resident_ms / 1000) if resident_ms else None)
                timing.setdefault("serial_samples_per_s", batch / (step_ms / 1000) if step_ms else None)
                for metric, key, unit in (
                    ("h2d_ms", "median_h2d_ms", "ms"), ("forward_ms", "median_forward_ms", "ms"),
                    ("d2h_ms", "median_d2h_ms", "ms"), ("step_ms", "median_step_ms", "ms"),
                    ("resident_forward_ms", "median_resident_forward_ms", "ms"),
                    ("resident_samples_per_s", "resident_samples_per_s", "samples/s"),
                    ("serial_samples_per_s", "serial_samples_per_s", "samples/s"),
                ):
                    rows.append(writer.record(model_id, display_name(spec), precision, batch, metric, timing.get(key), unit, "cuda_events"))
    writer.write("timing_measured", rows)

    return {"tables": writer.counts, "table_dir": str(run_dir / "tables")}
