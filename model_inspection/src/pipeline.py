"""Pipeline stages: measurement, NCU, Nsight Systems, and report generation.

The default CLI orchestration runs each stage as its own process so no parent
CUDA context can block the NCU/Nsys child processes on Exclusive_Process GPUs.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

from catalog import flatten_tensors, load_catalog, make_inputs, resolve_artifact, schema_bytes, spec_for, tensor_bytes
import latex
import ncu
import nsys
import plots
import tables


def _torch():
    import torch

    return torch


def run_measure_stage(catalog_path: Path, hardware_path: Path, models: list[str], precision: str,
                      memory_fraction: float, batch_step: int, max_batch: int | None, manual_ladder: list[int] | None,
                      warmup: int, iterations: int, device_str: str, run_dir: Path, logger) -> dict:
    torch = _torch()
    import inspection

    catalog = load_catalog(catalog_path)
    hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else {}
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    inspection.configure_precision(precision)
    run = {
        "schema_version": 2,
        "run_meta": {
            "timestamp": datetime.now().isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "precision_policy": precision,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32 if device.type == "cuda" else False,
            "catalog": str(catalog_path),
            "hardware_reference": str(hardware_path),
            "hbm_reference_gbps": hardware.get("hbm_gbps", 3938.0),
            "compute_gflops": hardware.get("compute_gflops", {"fp32": 60320.0, "tf32": 482600.0}),
            "gpu": inspection.gpu_info(device),
            "link_calibration": inspection.calibrate_link(device, 64 * 1024**2, 20),
            "adaptive_batch_discovery": {
                "memory_fraction": memory_fraction,
                "batch_step": batch_step,
                "max_batch": max_batch,
                "manual_ladder": manual_ladder,
            },
        },
        "models": [],
    }
    for model_id in models:
        spec = spec_for(catalog, model_id)
        artifact = resolve_artifact(catalog_path.parent, spec, device)
        logger.info(f"[{model_id}] loading {artifact}")
        model = torch.jit.load(str(artifact), map_location=device).eval()
        actual_inputs = make_inputs(spec, 1, device)
        with torch.inference_mode():
            actual_outputs = flatten_tensors(model(*actual_inputs))
        observed = (tensor_bytes(list(actual_inputs)), tensor_bytes(actual_outputs))
        contract = (schema_bytes(spec["inputs"]), schema_bytes(spec["outputs"]))
        if observed != contract:
            raise RuntimeError(f"catalog I/O contract disagrees with executed {model_id}: observed {observed}, catalog {contract}")
        logger.info("  adaptive batch discovery")
        discovery = inspection.discover_batches(model, spec, device, memory_fraction=memory_fraction,
                                                batch_step=batch_step, max_batch=max_batch, manual=manual_ladder,
                                                logger=logger)
        ladder = discovery["ladder"]
        if not ladder:
            raise RuntimeError(f"adaptive discovery produced an empty ladder for {model_id}")
        logger.info(f"  ladder: {ladder} (stop: {discovery['stop_reason']})")
        record = {
            "model_id": model_id,
            "display_name": spec.get("display_name", model_id),
            "artifact": {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": inspection.sha256_file(artifact)},
            "observed_io": {
                "input_bytes_per_sample": observed[0], "output_bytes_per_sample": observed[1],
                "catalog_input_bytes_per_sample": contract[0], "catalog_output_bytes_per_sample": contract[1],
            },
            "runtime_graph": inspection.runtime_graph_audit(),
            "weights": inspection.storage_summary(model),
            "adaptive": discovery,
            "workspace": inspection.measure_workspace(model, actual_inputs),
            "batches": [],
        }
        for batch in ladder:
            logger.info(f"  B={batch}")
            timing = inspection.benchmark_cuda(model, spec, batch, device, warmup, iterations)
            record["batches"].append(timing)
        run["models"].append(record)
        del model
        import gc

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    destination = run_dir / f"measurement_{precision}.json"
    destination.write_text(json.dumps(run, indent=2))
    logger.info(f"Results written to {destination}")
    return run


def resolve_ncu_batches(run_dir: Path, models: list[str], precisions: list[str],
                        spec: str = "1,1000", max_batch: int | None = None) -> dict[str, list[int]]:
    """Resolve the NCU batch selection.

    ``spec`` is either a comma-separated list of integers or ``ladder``/``full``
    to profile every batch of each model's discovered adaptive ladder. Requested
    batches are snapped to the nearest discovered ladder batch, so NCU, timing,
    and tables always share the same batch grid (for example 1000 -> 1024 when
    the ladder doubles).
    """
    ladders: dict[str, list[int]] = {}
    for precision in precisions:
        path = run_dir / f"measurement_{precision}.json"
        if not path.exists():
            continue
        for record in json.loads(path.read_text())["models"]:
            ladder = record.get("adaptive", {}).get("ladder") or [timing["batch_size"] for timing in record["batches"]]
            ladders[record["model_id"]] = sorted({int(value) for value in ladder})
    on_ladder = spec.strip().lower() in ("ladder", "full")
    requested = [] if on_ladder else [int(value) for value in spec.split(",") if value.strip()]
    targets: dict[str, list[int]] = {}
    for model in models:
        ladder = ladders.get(model) or [1]
        if on_ladder:
            batches = set(ladder)
        else:
            batches = {min(ladder, key=lambda candidate: (abs(candidate - value), -candidate)) for value in requested}
        if max_batch is not None:
            batches = {value for value in batches if value <= max_batch}
        targets[model] = sorted(batches)
    return targets


def run_ncu_stage(catalog_path: Path, models: list[str], precisions: list[str], run_dir: Path, ncu_bin: Path,
                  metrics: list[str], timeout: int, logger, allow_missing: bool,
                  batches_spec: str = "1,1000", max_batch: int | None = None,
                  cache_control: str = "none", replay_mode: str = "range", clock_control: str = "none") -> tuple[list[dict], bool]:
    batches_by_model = resolve_ncu_batches(run_dir, models, precisions, batches_spec, max_batch)
    logger.info(f"NCU batches per model: {batches_by_model} (cache={cache_control}, replay={replay_mode})")
    entries = ncu.collect(catalog_path, models, precisions, batches_by_model, ncu_bin, metrics, run_dir, timeout, logger,
                          cache_control=cache_control, replay_mode=replay_mode, clock_control=clock_control)
    version = ncu.ncu_version(ncu_bin)
    summary_path = run_dir / "ncu_summary.json"
    previous: dict[tuple[str, str, int], dict] = {}
    if summary_path.exists():
        old = json.loads(summary_path.read_text())
        previous = {(entry["model"], entry["precision"], int(entry["batch"])): entry for entry in old.get("entries", [])}
    measured = set(models)
    merged = {key: entry for key, entry in previous.items() if key[0] not in measured}
    merged.update({(entry["model"], entry["precision"], int(entry["batch"])): entry for entry in entries})
    summary = {
        "schema_version": 2, "created": datetime.now().isoformat(), "ncu_bin": str(ncu_bin),
        "ncu_version": version, "metrics": ",".join(metrics),
        "cache_control": cache_control, "replay_mode": replay_mode, "clock_control": clock_control,
        "batches_spec": batches_spec, "entries": [merged[key] for key in sorted(merged)],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    missing = [entry for entry in entries if not ncu.entry_ok(entry)]
    strict_ok = not missing or allow_missing
    logger.info(f"NCU entries: {len(entries)}, missing: {len(missing)}; summary written")
    return entries, strict_ok


def run_nsys_stage(catalog_path: Path, catalog: dict, models: list[str], precisions: list[str], run_dir: Path,
                   nsys_bin: str, repeats: int, timeout: int, logger, batches_spec: str = "1,1000") -> list[dict]:
    batches_by_model = resolve_ncu_batches(run_dir, models, precisions, batches_spec)
    logger.info(f"Nsys batches per model: {batches_by_model}")
    entries = nsys.collect(catalog_path, catalog, models, precisions, batches_by_model, nsys_bin, repeats, run_dir, timeout, logger)
    summary = {"schema_version": 2, "created": datetime.now().isoformat(), "nsys_bin": nsys_bin, "entries": entries}
    (run_dir / "nsys_summary.json").write_text(json.dumps(summary, indent=2))
    ok = sum(1 for entry in entries if entry.get("h2d_count") or entry.get("d2h_count"))
    logger.info(f"Nsys entries: {len(entries)}, with memcpy records: {ok}; summary written")
    return entries


def run_report_stage(run_dir: Path, catalog_path: Path, hardware_path: Path, models: list[str], precisions: list[str],
                     comparison_batch: int, emit_latex: bool, compile_latex: bool, latexmk: str | None,
                     logger, texlive_bin: str | None = None) -> dict:
    catalog = load_catalog(catalog_path)
    hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else {}
    measurements = {precision: json.loads((run_dir / f"measurement_{precision}.json").read_text()) for precision in precisions}
    ncu_summary = json.loads((run_dir / "ncu_summary.json").read_text()) if (run_dir / "ncu_summary.json").exists() else {"entries": []}
    nsys_summary = json.loads((run_dir / "nsys_summary.json").read_text()) if (run_dir / "nsys_summary.json").exists() else {"entries": []}
    table_meta = tables.build_all(run_dir, catalog, models, precisions, measurements, ncu_summary, nsys_summary)
    logger.info(f"Tables: {table_meta['tables']}")

    flops = tables.read_csv(run_dir / "tables" / "flops_measured_hardware.csv")
    precision_ops = tables.read_csv(run_dir / "tables" / "precision_ops_measured.csv")
    hbm = tables.read_csv(run_dir / "tables" / "hbm3_measured.csv")
    pcie = tables.read_csv(run_dir / "tables" / "pcie_measured.csv")
    cache = tables.read_csv(run_dir / "tables" / "cache_l2_measured.csv")
    oi = tables.read_csv(run_dir / "tables" / "oi_hbm3_measured.csv")
    timing = tables.read_csv(run_dir / "tables" / "timing_measured.csv")

    def _batches(table: list[dict], metric: str, model: str) -> set[int]:
        return {int(row["batch_size"]) for row in table
                if row["model_id"] == model and row["metric"] == metric and row["value"] not in ("", None)}

    shared = [_batches(flops, "measured_flops", model) & _batches(timing, "resident_forward_ms", model)
              for model in models]
    intersection = set.intersection(*shared) if shared else set()
    candidates = intersection or _batches(timing, "resident_forward_ms", models[0]) or {comparison_batch}
    effective_batch = min(candidates, key=lambda candidate: (abs(candidate - comparison_batch), candidate))
    if effective_batch != comparison_batch:
        logger.info(f"Comparison batch {comparison_batch} is not available for all models; using {effective_batch}")
    comparison_batch = effective_batch

    def missing_models(table: list[dict], metric: str) -> list[str]:
        have = {row["model_id"] for row in table
                if row["metric"] == metric and row["value"] not in ("", None)
                and int(row["batch_size"]) == comparison_batch}
        return [model for model in models if model not in have]

    for table_name, table, metric in (
        ("flops_measured_hardware", flops, "measured_flops"),
        ("precision_ops_measured", precision_ops, "total_flops"),
        ("hbm3_measured", hbm, "total"),
        ("pcie_measured", pcie, "total"),
        ("oi_hbm3_measured", oi, "operational_intensity"),
    ):
        missing = missing_models(table, metric)
        if missing:
            logger.warning(f"figure metric {metric} ({table_name}) missing models at B={comparison_batch}: {missing}")

    include_tf32 = any(row["metric"] == tables.TENSOR and row["value"] not in ("", None) and float(row["value"]) > 0
                       for row in flops)
    figures = []
    for function, args in (
        (plots.bar_measured, (run_dir, flops, "measured_flops", "flops", "FLOPs", "FLOPs", "bar_flops", comparison_batch)),
        (plots.bar_precision, (run_dir, precision_ops, comparison_batch)),
        (plots.bar_measured, (run_dir, hbm, "total", "bytes", "HBM3 RAM", "Bytes", "bar_hbm3", comparison_batch)),
        (plots.bar_measured, (run_dir, pcie, "total", "bytes", "PCIe", "Bytes", "bar_pcie", comparison_batch)),
        (plots.roofline_measured, (run_dir, oi, flops, timing, hardware, include_tf32)),
        (plots.l2_cache_bars, (run_dir, cache, comparison_batch)),
        (plots.hbm_vs_batch, (run_dir, hbm)),
        (plots.oi_vs_batch, (run_dir, oi)),
    ):
        path = function(*args)
        if path:
            figures.append(str(path))
    for precision in precisions:
        path = plots.throughput(run_dir, timing, precision, hardware)
        if path:
            figures.append(str(path))
    logger.info(f"Figures: {len(figures)}")

    provenance = {
        "run_name": run_dir.name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hostname": measurements[precisions[0]]["run_meta"]["hostname"],
        "gpu": measurements[precisions[0]]["run_meta"].get("gpu", {}),
        "ncu_version": ncu_summary.get("ncu_version", "unknown"),
        "ncu_cache_control": ncu_summary.get("cache_control", "unknown"),
        "ncu_replay_mode": ncu_summary.get("replay_mode", "unknown"),
        "comparison_batch": comparison_batch,
        "ladders": {record["model_id"]: record.get("adaptive", {}).get("ladder", [timing["batch_size"] for timing in record["batches"]])
                    for record in measurements[precisions[0]]["models"]},
        "nsys_collected": bool(nsys_summary.get("entries")),
        "figures": figures,
        "tables": table_meta["tables"],
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    result = {"figures": figures, "tables": table_meta["tables"], "latex": None, "pdf": None}
    if emit_latex:
        tex_path = latex.render(run_dir, catalog, models, precisions, hardware, comparison_batch, provenance)
        result["latex"] = str(tex_path)
        logger.info(f"LaTeX written to {tex_path}")
        if compile_latex:
            pdf = latex.compile_pdf(tex_path.parent, latexmk, logger, texlive_bin=texlive_bin)
            result["pdf"] = str(pdf) if pdf else None
    return result
