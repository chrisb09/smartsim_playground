#!/usr/bin/env python3
"""Repeat-measurement statistics at B=1 and ~80% VRAM.

For every model/precision this script determines a batch whose predicted peak
VRAM is close to a fraction (default 80%) of device memory, rounds it to the
nearest single-significant-digit value, verifies that it fits, and then repeats
the CUDA-event benchmark and the Nsight Compute counter pass N times. It reports
median, mean, sample standard deviation, min, max, interquartile range, and the
coefficient of variation for every measured value.

Examples:
  python measure_stats.py --skip-ncu --repeats 3 --models mmcp_test_mlp_m5 --run-name stats_smoke
  python measure_stats.py --fit-from-run generated/slurm_3983194 --repeats 10 --run-name stats_campaign
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import platform
import signal
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

import catalog
import inspection
import ncu
import nsys
import tables

DEFAULT_NCU_BIN = "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.8.0/bin/ncu"
TIMING_METRICS = (
    "median_h2d_ms", "median_forward_ms", "median_d2h_ms", "median_transfer_ms",
    "median_step_ms", "median_resident_forward_ms",
    "resident_samples_per_s", "serial_samples_per_s",
)
VRAM_METRICS = ("peak_allocated_bytes", "peak_reserved_bytes")
CSV_HEADER = ["model_id", "display_name", "precision", "batch_size", "metric", "unit",
              "n", "median", "mean", "std", "min", "max", "p25", "p75", "cv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Outputs are written to <output-dir>/<run-name>/ as stats.json, stats.csv, and per-repetition raw NCU CSVs.",
    )
    parser.add_argument("--catalog", default=str(ROOT / "model_catalog.json"), help="Model catalog JSON (default: canonical catalog)")
    parser.add_argument("--models", default="all", help="Comma-separated model IDs or 'all'")
    parser.add_argument("--precisions", default="fp32", help="Comma-separated precision policies (default: fp32)")
    parser.add_argument("--repeats", type=int, default=10, help="Repetitions per batch for timing and NCU (default: 10)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup forwards per timing repetition (default: 10)")
    parser.add_argument("--iterations", type=int, default=30, help="Timed iterations per phase per timing repetition (default: 30)")
    parser.add_argument("--memory-fraction", type=float, default=0.8, help="Target peak-VRAM fraction for the large batch (default: 0.8)")
    parser.add_argument("--ladder", default="x8", help="Ladder multiplier used only when probing VRAM (default: x8)")
    parser.add_argument("--max-batch", type=int, default=None, help="Optional cap for the large batch (useful for quick tests)")
    parser.add_argument("--fit-from-run", default=None, help="Run directory or measurement JSON to take the VRAM fit from (default: newest matching measurement)")
    parser.add_argument("--probe", action="store_true", help="Ignore stored VRAM fits and probe now")
    parser.add_argument("--resume", action="store_true", help="Continue an existing run directory, skipping batches that already have the requested repetitions")
    parser.add_argument("--rebatch", default=None, help="Comma-separated model IDs whose stored large batch is re-selected (with --resume); keeps completed batches")
    parser.add_argument("--skip-ncu", action="store_true", help="Skip the repeated Nsight Compute pass")
    parser.add_argument("--ncu-bin", default=DEFAULT_NCU_BIN, help="Working NCU binary (>= 2025.1 on driver 580.x)")
    parser.add_argument("--ncu-timeout", type=int, default=1800, help="Per-invocation NCU timeout (seconds)")
    parser.add_argument("--ncu-cache-control", choices=("none", "all"), default="none", help="NCU cache control (default: none, warm caches)")
    parser.add_argument("--ncu-replay-mode", choices=("range", "kernel", "application"), default="application", help="NCU replay mode (default: application)")
    parser.add_argument("--ncu-clock-control", choices=("none", "base", "reset"), default="none", help="NCU clock control (default: none)")
    parser.add_argument("--skip-nsys", action="store_true", help="Skip the Nsight Systems PCIe pass")
    parser.add_argument("--nsys-bin", default="nsys", help="Nsight Systems CLI binary")
    parser.add_argument("--nsys-repeats", type=int, default=12, help="Copy repeats per Nsight Systems trace (default: 12)")
    parser.add_argument("--nsys-timeout", type=int, default=1200, help="Per-invocation Nsight Systems timeout (seconds)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(ROOT / "generated"), help="Parent directory for run folders")
    parser.add_argument("--run-name", default=None, help="Run folder name (default: stats_<timestamp>)")
    return parser.parse_args()


def build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("measure_stats")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "pipeline.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def ladder_step(spec: str) -> int:
    text = spec.strip().lower()
    if text.startswith("x"):
        text = text[1:]
    try:
        value = int(text)
    except ValueError:
        raise SystemExit(f"invalid --ladder '{spec}': expected x<N>, for example x8")
    if value < 2:
        raise SystemExit(f"invalid --ladder '{spec}': the multiplier must be at least 2")
    return value


def round_down_sig_fig(value: float, digits: int = 2) -> int:
    if value < 1:
        return 1
    exponent = math.floor(math.log10(value))
    scale = 10 ** max(0, exponent - digits + 1)
    return max(1, int(value // scale) * scale)


def lower_sig_fig(value: float) -> int:
    if value <= 1:
        return 1
    scale = 10 ** math.floor(math.log10(value))
    leading = int(value // scale)
    if leading > 1:
        return max(1, int((leading - 1) * scale))
    return max(1, int(0.9 * scale))


def step_down(batch: int) -> int:
    return max(1, min(lower_sig_fig(batch), batch - 1))


def stored_fit(source: str | None, output_dir: Path, catalog_path: Path, model_id: str, precision: str) -> tuple[dict | None, str | None]:
    if source:
        path = Path(source).expanduser()
        paths = sorted(path.glob(f"measurement_{precision}.json")) if path.is_dir() else [path]
        strict = False
    else:
        paths = sorted(output_dir.glob(f"*/measurement_{precision}.json"), key=lambda candidate: candidate.stat().st_mtime, reverse=True)
        strict = True
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if strict:
            reference = data.get("run_meta", {}).get("catalog")
            if reference and Path(reference).resolve() != catalog_path.resolve():
                continue
        for record in data.get("models", []):
            if record.get("model_id") != model_id:
                continue
            fit = record.get("adaptive", {}).get("fit", {})
            if fit.get("slope_bytes_per_sample", 0) > 0 and "intercept_bytes" in fit:
                return fit, str(path)
    return None, None


def probe_fit(model, spec, device, step, max_batch, logger) -> dict | None:
    discovery = inspection.discover_batches(model, spec, device, memory_fraction=0.9, batch_step=step,
                                            max_batch=max_batch, logger=logger)
    fit = discovery.get("fit", {})
    logger.info(f"  probed ladder: {discovery['ladder']} (stop: {discovery['stop_reason']})")
    return fit if fit.get("slope_bytes_per_sample", 0) > 0 else None


def timing_probe(model, spec, batch, device) -> int | None:
    """Peak device memory of the timing workload (retains the previous output)."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    inspection.benchmark_cuda(model, spec, batch, device, warmup=2, iterations=1)
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    torch.cuda.empty_cache()
    return peak


STEP_DOWN_LIMIT = 12
REFINE_PROBE_LIMIT = 16


def refine_batch(model, spec, device, low, high, gate, logger, relative_tolerance: float = 0.01) -> int:
    """Bisect the largest gate-feasible batch between a passing and a failing value."""
    best = low
    probes = 0
    while high - low > max(1, int(relative_tolerance * best)) and probes < REFINE_PROBE_LIMIT:
        probes += 1
        middle = (low + high) // 2
        try:
            peak = timing_probe(model, spec, middle, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            high = middle
            continue
        if peak is None or peak <= gate:
            low = best = middle
        else:
            high = middle
    logger.info(f"  refined feasible maximum B={best:,} (bisection probes {probes}, failed above {high:,})")
    return max(1, round_down_sig_fig(best))


def choose_batch(model, spec, device, fit, fraction, max_batch, logger) -> tuple[int, int | None]:
    total = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    gate = int(fraction * total)
    target = (gate - fit["intercept_bytes"]) / fit["slope_bytes_per_sample"]
    batch = round_down_sig_fig(target)
    if max_batch is not None:
        batch = min(batch, max_batch)
    logger.info(f"  B{fraction:.0%} target {target:,.0f} -> candidate B={batch} (gate {gate / 2**30:.2f} GiB)")
    failing: int | None = None
    passing: tuple[int, int | None] | None = None
    steps = 0
    while batch > 1 and steps < STEP_DOWN_LIMIT:
        steps += 1
        try:
            peak = timing_probe(model, spec, batch, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            logger.warning(f"  B={batch} failed ({type(exc).__name__}: {exc}); stepping down")
            torch.cuda.empty_cache()
            failing = batch
            batch = step_down(batch)
            continue
        if peak is None or peak <= gate:
            if peak is not None:
                logger.info(f"  verified B={batch}: {peak / 2**30:.2f} GiB <= {gate / 2**30:.2f} GiB gate (timing workload)")
            else:
                logger.info(f"  using B={batch} (no CUDA memory gate on {device.type})")
            passing = (batch, peak)
            break
        logger.info(f"  B={batch} peaks at {peak / 2**30:.2f} GiB > gate; stepping down")
        failing = batch
        batch = step_down(batch)
    if passing is None:
        if steps >= STEP_DOWN_LIMIT:
            logger.warning(f"  step-down limit ({STEP_DOWN_LIMIT}) reached at B={batch}")
        try:
            return 1, timing_probe(model, spec, 1, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            return 1, None
    if failing is not None and failing > passing[0]:
        refined = refine_batch(model, spec, device, passing[0], failing, gate, logger)
        if refined > passing[0]:
            try:
                peak = timing_probe(model, spec, refined, device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()
                peak = None
            if peak is None or peak <= gate:
                suffix = f": {peak / 2**30:.2f} GiB <= {gate / 2**30:.2f} GiB gate" if peak is not None else ""
                logger.info(f"  refined B={refined} verified{suffix}")
                passing = (refined, peak)
    return passing


def timing_reps(model, spec, batch, device, warmup, iterations, repeats, logger) -> list[dict]:
    records = []
    for rep in range(1, repeats + 1):
        timing = inspection.benchmark_cuda(model, spec, batch, device, warmup, iterations)
        vram = inspection.measure_vram(model, spec, batch, device)
        records.append({"rep": rep, "timing": timing, "vram": vram})
        if rep == 1 or rep == repeats or rep % 5 == 0:
            logger.info(f"    timing rep {rep}/{repeats}: forward {timing['median_forward_ms']:.4f} ms, "
                        f"resident {timing['median_resident_forward_ms']:.4f} ms")
    return records


def ncu_reps(catalog_path, model_id, precision, batch, args, stats_dir, repeats, logger) -> list[dict]:
    records = []
    for rep in range(1, repeats + 1):
        entries = ncu.collect(catalog_path, [model_id], [precision], {model_id: [batch]}, Path(args.ncu_bin),
                              ncu.DEFAULT_METRICS, stats_dir / f"rep{rep:02d}", args.ncu_timeout, logger,
                              cache_control=args.ncu_cache_control, replay_mode=args.ncu_replay_mode,
                              clock_control=args.ncu_clock_control)
        if entries:
            records.append(entries[0])
    return records


def ncu_metrics(entry: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    counters = entry.get("counters", {})
    for name, value in counters.items():
        if value not in (None, ncu.MISSING):
            values[name] = float(value)
    flops = tables.measured_flops(counters)
    dram = float(entry.get("dram_bytes_total") or 0.0)
    l2 = float(entry.get("l2_bytes_total") or 0.0)
    values["kernel_count"] = float(entry.get("kernel_count", 0))
    values["dram_bytes_total"] = dram
    values["dram_bytes_read"] = float(entry.get("dram_bytes_read") or 0.0)
    values["dram_bytes_write"] = float(entry.get("dram_bytes_write") or 0.0)
    values["l2_bytes_total"] = l2
    values["l2_bytes_read"] = float(entry.get("l2_bytes_read") or 0.0)
    values["l2_bytes_write"] = float(entry.get("l2_bytes_write") or 0.0)
    if flops is not None:
        values["measured_flops"] = flops
    if flops and dram:
        values["oi_flops_per_byte"] = flops / dram
    if dram and l2:
        values["l2_over_dram"] = l2 / dram
    return values


def nsys_once(catalog_path, catalog_data, model_id, precision, batch, args, run_dir, logger) -> dict | None:
    entries = nsys.collect(catalog_path, catalog_data, [model_id], [precision], {model_id: [batch]},
                           args.nsys_bin, args.nsys_repeats, run_dir, args.nsys_timeout, logger)
    return entries[0] if entries else None


def pcie_metrics(entry: dict) -> dict[str, float]:
    repeats = entry.get("repeats") or 1
    values: dict[str, float] = {}
    for key in ("h2d_bytes", "d2h_bytes"):
        value = entry.get(key)
        if value is not None:
            values[f"pcie_{key}"] = float(value) / repeats
    if "pcie_h2d_bytes" in values and "pcie_d2h_bytes" in values:
        values["pcie_total_bytes"] = values["pcie_h2d_bytes"] + values["pcie_d2h_bytes"]
    return values


def metric_unit(metric: str) -> str:
    if metric.endswith("_ms"):
        return "ms"
    if metric.endswith("samples_per_s"):
        return "samples/s"
    if metric == "kernel_count":
        return "kernels"
    if metric in ("l2_over_dram", "oi_flops_per_byte"):
        return "ratio"
    if metric.endswith("_bytes") or metric.startswith(("dram__", "lts__")):
        return "bytes"
    if metric.startswith("smsp__sass_thread_inst_executed_op"):
        return "instructions"
    if metric.startswith("sm__ops_path") or metric.endswith("_flops") or metric == "measured_flops":
        return "ops"
    return "value"


def summarize(values: list[float]) -> dict | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None
    summary = {
        "n": len(clean),
        "median": statistics.median(clean),
        "mean": statistics.fmean(clean),
        "min": min(clean),
        "max": max(clean),
        "p25": float(np.percentile(clean, 25)),
        "p75": float(np.percentile(clean, 75)),
    }
    summary["std"] = statistics.stdev(clean) if len(clean) > 1 else 0.0
    summary["cv"] = summary["std"] / summary["mean"] if summary["mean"] else None
    return summary


def write_outputs(results: dict, run_dir: Path) -> tuple[Path, Path]:
    stats_path = run_dir / "stats.json"
    stats_path.write_text(json.dumps(results, indent=2))
    csv_path = run_dir / "stats.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for record in results["models"]:
            for batch_record in record["batches"]:
                for metric, summary in sorted(batch_record["stats"].items()):
                    writer.writerow([record["model_id"], record["display_name"], record["precision"], batch_record["batch"],
                                     metric, metric_unit(metric), summary["n"], summary["median"], summary["mean"],
                                     summary["std"], summary["min"], summary["max"], summary["p25"], summary["p75"], summary["cv"]])
    return stats_path, csv_path


def log_model_summary(record: dict, logger) -> None:
    logger.info(f"{record['model_id']} B80={record['large_batch']['batch']} fit_source={record['fit_source']}")
    for batch_record in record["batches"]:
        stats = batch_record["stats"]
        resident = stats.get("median_resident_forward_ms", {})
        step_stats = stats.get("median_step_ms", {})
        dram = stats.get("dram_bytes_total", {})
        flops = stats.get("measured_flops", {})
        cv = resident.get("cv")
        suffix = f" (cv {cv:.2%})" if cv is not None else ""
        logger.info(f"  B={batch_record['batch']:<9d} resident {resident.get('median', float('nan')):.4f} ms{suffix}")
        logger.info(f"    step {step_stats.get('median', float('nan')):.4f} ms, "
                    f"DRAM {dram.get('median', float('nan')) / 1e9:.3f} GB, "
                    f"FLOPs {flops.get('median', float('nan')):.4e}")
        pcie = batch_record.get("pcie") or {}
        if pcie.get("h2d_bytes") is not None or pcie.get("d2h_bytes") is not None:
            repeats = pcie.get("repeats") or 1
            h2d = (pcie.get("h2d_bytes") or 0) / repeats
            d2h = (pcie.get("d2h_bytes") or 0) / repeats
            logger.info(f"    PCIe H2D {h2d:,.0f} B + D2H {d2h:,.0f} B = {h2d + d2h:,.0f} B per inference")


def main() -> int:
    args = parse_args()
    catalog_path = Path(args.catalog).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    run_dir = output_dir / (args.run_name or f"stats_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but unavailable")
        return 2
    catalog_data = catalog.load_catalog(catalog_path)
    models = catalog.select_models(catalog_data, args.models)
    precisions = [value.strip() for value in args.precisions.split(",") if value.strip()]
    step = ladder_step(args.ladder)
    if args.repeats < 1:
        logger.error("--repeats must be at least 1")
        return 2
    ncu_runs = 0 if args.skip_ncu else len(models) * len(precisions) * 2 * args.repeats
    rebatch = {value.strip() for value in (args.rebatch or "").split(",") if value.strip()}
    logger.info(f"models={models} precisions={precisions} repeats={args.repeats} "
                f"memory_fraction={args.memory_fraction} ncu_invocations~{ncu_runs} rebatch={sorted(rebatch)}")

    results = {
        "schema_version": 1,
        "run_meta": {
            "timestamp": datetime.now().isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": inspection.gpu_info(device),
            "catalog": str(catalog_path),
            "models": models,
            "precisions": precisions,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "memory_fraction": args.memory_fraction,
            "max_batch": args.max_batch,
            "fit_from_run": args.fit_from_run,
            "probe": args.probe,
            "rebatch": sorted(rebatch),
            "skip_ncu": args.skip_ncu,
            "ncu_bin": args.ncu_bin,
            "ncu_cache_control": args.ncu_cache_control,
            "ncu_replay_mode": args.ncu_replay_mode,
            "ncu_clock_control": args.ncu_clock_control,
            "skip_nsys": args.skip_nsys,
            "nsys_bin": args.nsys_bin,
            "nsys_repeats": args.nsys_repeats,
            "flop_convention": tables.FLOP_CONVENTION,
        },
        "models": [],
    }
    completed: set[tuple[str, str, int]] = set()
    if args.resume:
        stats_file = run_dir / "stats.json"
        if stats_file.exists():
            results = json.loads(stats_file.read_text())
            results.setdefault("run_meta", {})["resumed_at"] = datetime.now().isoformat()
            for record in results.get("models", []):
                model_precision = record.get("precision", precisions[0] if precisions else "fp32")
                for batch_record in record.get("batches", []):
                    timing_ok = len(batch_record.get("timing", [])) >= args.repeats
                    ncu_ok = args.skip_ncu or len(batch_record.get("ncu", [])) >= args.repeats
                    pcie_ok = args.skip_nsys or batch_record.get("pcie") is not None
                    if timing_ok and ncu_ok and pcie_ok:
                        completed.add((record["model_id"], model_precision, int(batch_record["batch"])))
            logger.info(f"resuming {stats_file}: {len(completed)} completed batches {sorted(completed)}")
        else:
            logger.warning(f"--resume given but {stats_file} does not exist; starting fresh")

    def flush_on_term(signum, frame):
        try:
            write_outputs(results, run_dir)
        finally:
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, flush_on_term)

    for precision in precisions:
        inspection.configure_precision(precision)
        for model_id in models:
            spec = catalog.spec_for(catalog_data, model_id)
            record = next((item for item in results["models"]
                           if item.get("model_id") == model_id and item.get("precision") == precision), None)
            stored_batch = int(record["large_batch"]["batch"]) if record and record.get("large_batch", {}).get("batch") else None
            if model_id in rebatch and stored_batch is not None:
                logger.info(f"[{model_id}/{precision}] rebatch: ignoring stored large batch B={stored_batch}")
                stored_batch = None
            if stored_batch is not None and all((model_id, precision, batch) in completed
                                                for batch in ([1] if stored_batch == 1 else [1, stored_batch])):
                logger.info(f"[{model_id}/{precision}] already complete; skipping")
                continue
            logger.info(f"[{model_id}/{precision}]")
            try:
                artifact = catalog.resolve_artifact(catalog_path.parent, spec, device)
            except FileNotFoundError as exc:
                logger.error(f"  no artifact: {exc}")
                continue
            model = torch.jit.load(str(artifact), map_location=device).eval()
            if stored_batch is not None:
                b80, vram_b80 = stored_batch, record["large_batch"].get("verified_peak_allocated_bytes")
                logger.info(f"  reusing stored B80={b80}")
            else:
                fit, provenance = (None, None) if args.probe else stored_fit(args.fit_from_run, output_dir, catalog_path, model_id, precision)
                if fit is None:
                    logger.info("  no usable stored fit; probing VRAM")
                    fit = probe_fit(model, spec, device, step, args.max_batch, logger)
                    provenance = "probe"
                if fit is None:
                    logger.error(f"  could not obtain a VRAM fit for {model_id}; skipping")
                    del model
                    continue
                b80, vram_b80 = choose_batch(model, spec, device, fit, args.memory_fraction, args.max_batch, logger)
            batches = [1] if b80 == 1 else [1, b80]
            large_batch = {"batch": b80, "verified_peak_allocated_bytes": vram_b80,
                           "verification": "benchmark_cuda peak (warmup 2, iterations 1)"}
            if record is None:
                record = {
                    "model_id": model_id,
                    "display_name": spec.get("short_name") or spec.get("display_name", model_id),
                    "precision": precision,
                    "artifact": {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": inspection.sha256_file(artifact)},
                    "fit": fit,
                    "fit_source": provenance,
                    "large_batch": large_batch,
                    "batches": [],
                }
                results["models"].append(record)
            else:
                record["large_batch"] = large_batch
                if stored_batch is None:
                    record["fit"] = fit
                    record["fit_source"] = provenance
            for batch in batches:
                existing = next((item for item in record["batches"] if int(item["batch"]) == batch), None)
                timing_done = existing is not None and len(existing.get("timing", [])) >= args.repeats
                ncu_done = args.skip_ncu or (existing is not None and len(existing.get("ncu", [])) >= args.repeats)
                pcie_done = args.skip_nsys or (existing is not None and existing.get("pcie") is not None)
                if timing_done and ncu_done and pcie_done:
                    logger.info(f"  B={batch} already complete; skipping")
                    continue
                logger.info(f"  B={batch}")
                if timing_done:
                    timing = existing["timing"]
                else:
                    timing = timing_reps(model, spec, batch, device, args.warmup, args.iterations, args.repeats, logger)
                if ncu_done:
                    ncu_entries = (existing or {}).get("ncu") or []
                elif args.skip_ncu:
                    ncu_entries = []
                else:
                    ncu_entries = ncu_reps(catalog_path, model_id, precision, batch, args, run_dir, args.repeats, logger)
                if pcie_done:
                    pcie = existing.get("pcie")
                elif args.skip_nsys:
                    pcie = None
                else:
                    pcie = nsys_once(catalog_path, catalog_data, model_id, precision, batch, args, run_dir, logger)
                values: dict[str, list[float]] = defaultdict(list)
                for rep_record in timing:
                    for key in TIMING_METRICS:
                        if rep_record["timing"].get(key) is not None:
                            values[key].append(rep_record["timing"][key])
                    for key in VRAM_METRICS:
                        if rep_record["vram"].get(key) is not None:
                            values[key].append(rep_record["vram"][key])
                    values["input_bytes"].append(rep_record["timing"]["input_bytes"])
                    values["output_bytes"].append(rep_record["timing"]["output_bytes"])
                for entry in ncu_entries:
                    for key, value in ncu_metrics(entry).items():
                        values[key].append(value)
                if pcie:
                    for key, value in pcie_metrics(pcie).items():
                        values[key].append(value)
                stats = {metric: summary for metric, summary in ((key, summarize(value)) for key, value in sorted(values.items())) if summary}
                record["batches"] = [item for item in record["batches"] if int(item["batch"]) != batch]
                record["batches"].append({"batch": batch, "timing": timing, "ncu": ncu_entries, "pcie": pcie, "stats": stats})
                record["batches"].sort(key=lambda item: int(item["batch"]))
                write_outputs(results, run_dir)
            log_model_summary(record, logger)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    stats_path, csv_path = write_outputs(results, run_dir)
    logger.info(f"stats written to {stats_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
