#!/usr/bin/env python3
"""Measurement-only model analysis: measure, collect counters, and report.

Runs measurement passes (see README) and emits CSV tables, figures, and an
optional LaTeX report:

1. ``measure``: per-model adaptive batch discovery (VRAM-gated), CUDA-event
   timings, storage/workspace inspection, and link calibration.
2. ``ncu``: Nsight Compute DRAM, L2, and hardware arithmetic counters with warm
   caches (``--cache-control none``) and application replay.
3. ``nsys``: one Nsight Systems memcpy trace per model at B=1 and B=1000.
4. ``report``: CSV tables, measured bar charts, DRAM/L2 figures, measured
   roofline, throughput, VRAM scaling, and the LaTeX report (CPU only; rerun
   with ``--from-run``).

Examples:
  # Login-node run on a free GPU (adaptive ladder to the 90% VRAM gate):
  CUDA_VISIBLE_DEVICES=GPU-8ff8d0c7-8d30-8e55-0980-ac69fc03a6b8 \\
  python analyze.py --models mmcp_test_mlp_m5 --run-name smoke

  # Profile every discovered ladder batch with NCU:
  python analyze.py --models watercnn --ncu-batches ladder --run-name ladder_test

  # Full campaign (submitted via Slurm):
  sbatch --exclusive run_analysis.sbatch

  # Regenerate tables/figures/report from an existing run (no GPU):
  python analyze.py --from-run generated/<run>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import latex  # noqa: E402
import pipeline  # noqa: E402

DEFAULT_NCU_BIN = "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.8.0/bin/ncu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages can be rerun individually with --stage (internal). Outputs are written to <output-dir>/<run-name>/.",
    )
    parser.add_argument("--catalog", default=str(ROOT / "model_catalog.json"), help="Model catalog JSON (default: canonical catalog)")
    parser.add_argument("--hardware", default=str(ROOT / "hardware_h100.json"), help="Hardware reference JSON with HBM/compute ceilings")
    parser.add_argument("--models", default="all", help="Comma-separated model IDs or 'all'")
    parser.add_argument("--precisions", default="fp32", help="Comma-separated precision policies (default: fp32; TF32 must be passed explicitly)")
    parser.add_argument("--memory-fraction", type=float, default=0.9, help="VRAM gate as a fraction of total device memory (default: 0.9)")
    parser.add_argument("--ladder", default="x10", help="Adaptive ladder multiplier as x<N>, e.g. x2, x5, x10; N must be at least 2 (default: x10)")
    parser.add_argument("--max-batch", type=int, default=None, help="Optional absolute batch cap for the adaptive ladder (default: none)")
    parser.add_argument("--batch-ladder", default=None, help="Manual comma-separated batch ladder; bypasses adaptive discovery (debug)")
    parser.add_argument("--comparison-batch", type=int, default=1000, help="Target batch size for comparison tables and figures; the closest batch available for all models is used (default: 1000)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(ROOT / "generated"), help="Parent directory for run folders")
    parser.add_argument("--run-name", default=None, help="Run folder name (default: run_<timestamp>)")
    parser.add_argument("--ncu-bin", default=DEFAULT_NCU_BIN, help="Working NCU binary (>= 2025.1 on driver 580.x)")
    parser.add_argument("--ncu-timeout", type=int, default=1800, help="Per-invocation NCU timeout (seconds)")
    parser.add_argument("--ncu-batches", default="1,1000", help="NCU batches: comma-separated batch sizes or 'ladder'/'full'; explicit values snap to the nearest discovered ladder batch (default: 1,1000)")
    parser.add_argument("--ncu-max-batch", type=int, default=None, help="Optional upper batch cap for NCU profiling")
    parser.add_argument("--ncu-cache-control", choices=("none", "all"), default="none", help="NCU cache control (default: none, warm caches)")
    parser.add_argument("--ncu-replay-mode", choices=("range", "kernel", "application"), default="application", help="NCU replay mode (default: application; range replay does not support SASS counters)")
    parser.add_argument("--ncu-clock-control", choices=("none", "base", "reset"), default="none", help="NCU clock control (default: none)")
    parser.add_argument("--allow-missing-ncu", action="store_true", help="Do not fail the pipeline when NCU entries are missing")
    parser.add_argument("--skip-ncu", action="store_true", help="Skip the NCU pass")
    parser.add_argument("--nsys-bin", default="nsys", help="Nsight Systems CLI binary")
    parser.add_argument("--nsys-repeats", type=int, default=12, help="Copy repeats per Nsight Systems trace")
    parser.add_argument("--nsys-timeout", type=int, default=1200, help="Per-invocation Nsight Systems timeout (seconds)")
    parser.add_argument("--skip-nsys", action="store_true", help="Skip the Nsight Systems pass")
    parser.add_argument("--no-latex", action="store_true", help="Do not generate the LaTeX report")
    parser.add_argument("--no-compile-latex", action="store_true", help="Generate report.tex but do not run latexmk")
    parser.add_argument("--latexmk", default=None, help="Path to latexmk (default: PATH or TeX Live 2026 install)")
    parser.add_argument("--texlive-bin", default=None, help="TeX installation bin directory to prepend to PATH when compiling (default: directory of the latexmk binary)")
    parser.add_argument("--from-run", default=None, help="Regenerate tables/figures/report from an existing run directory (no GPU)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    parser.add_argument("--stage", choices=("measure", "ncu", "nsys", "report"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precision-arg", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("analyze")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "pipeline.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def measured_models(catalog_path: Path, requested: str) -> list[str]:
    from catalog import load_catalog, select_models

    return select_models(load_catalog(catalog_path), requested)


def manual_ladder(args: argparse.Namespace) -> list[int] | None:
    if not args.batch_ladder:
        return None
    return [int(value) for value in args.batch_ladder.split(",") if value.strip()]


def ladder_step(spec: str) -> int:
    """Parse a named ladder multiplier such as ``x10`` into its integer step."""
    text = spec.strip().lower()
    if text.startswith("x"):
        text = text[1:]
    try:
        value = int(text)
    except ValueError:
        raise SystemExit(f"invalid --ladder '{spec}': expected x<N>, for example x10")
    if value < 2:
        raise SystemExit(f"invalid --ladder '{spec}': the multiplier must be at least 2 (x2 or higher)")
    return value


def preflight(args: argparse.Namespace, catalog_path: Path, hardware_path: Path, models: list[str],
              logger: logging.Logger, stage: str | None) -> list[str]:
    """Validate inputs and tools for the requested stage; returns error messages."""
    problems: list[str] = []
    if not catalog_path.exists():
        problems.append(f"model catalog not found: {catalog_path}")
    else:
        try:
            from catalog import load_catalog

            load_catalog(catalog_path)
        except Exception as exc:
            problems.append(f"invalid model catalog {catalog_path}: {exc}")
    if not hardware_path.exists():
        logger.warning(f"hardware reference not found: {hardware_path}; using built-in fallbacks")
    if stage in (None, "measure") and catalog_path.exists():
        try:
            import torch
            from catalog import load_catalog, resolve_artifact, spec_for

            catalog = load_catalog(catalog_path)
            device = torch.device(args.device)
            for model in models:
                try:
                    resolve_artifact(catalog_path.parent, spec_for(catalog, model), device)
                except FileNotFoundError as exc:
                    problems.append(str(exc))
                except StopIteration:
                    problems.append(f"model '{model}' is not defined in {catalog_path}")
        except Exception as exc:
            problems.append(f"cannot resolve model artifacts from {catalog_path}: {exc}")
    if stage in (None, "ncu") and not args.skip_ncu:
        ncu_path = Path(args.ncu_bin).expanduser()
        if not ncu_path.exists():
            problems.append(f"Nsight Compute binary not found: {ncu_path} (override with --ncu-bin)")
    if stage in (None, "nsys") and not args.skip_nsys:
        if not (shutil.which(args.nsys_bin) or Path(args.nsys_bin).expanduser().exists()):
            problems.append(f"Nsight Systems binary not found: {args.nsys_bin} (override with --nsys-bin)")
    if stage in (None, "report") and not args.no_latex and not args.no_compile_latex:
        if args.latexmk and not Path(args.latexmk).expanduser().exists():
            problems.append(f"latexmk not found: {Path(args.latexmk).expanduser()} (override with --latexmk)")
        elif not args.latexmk and latex.find_latexmk(None) is None:
            logger.warning("latexmk not found; report will be generated without a compiled PDF")
        if args.texlive_bin and not Path(args.texlive_bin).expanduser().is_dir():
            problems.append(f"TeX bin directory not found: {Path(args.texlive_bin).expanduser()}")
    return problems


def config_for(args: argparse.Namespace) -> dict:
    return {
        "catalog": str(Path(args.catalog).expanduser()),
        "hardware": str(Path(args.hardware).expanduser()),
        "models": args.models,
        "precisions": args.precisions,
        "memory_fraction": args.memory_fraction,
        "ladder": args.ladder,
        "batch_step": ladder_step(args.ladder),
        "max_batch": args.max_batch,
        "batch_ladder": args.batch_ladder,
        "comparison_batch": args.comparison_batch,
        "ncu_batches": args.ncu_batches,
        "ncu_max_batch": args.ncu_max_batch,
        "ncu_cache_control": args.ncu_cache_control,
        "ncu_replay_mode": args.ncu_replay_mode,
        "ncu_clock_control": args.ncu_clock_control,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "device": args.device,
    }


def main() -> int:
    args = parse_args()

    if args.from_run:
        run_dir = Path(args.from_run).expanduser().resolve()
        if not run_dir.exists():
            print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
            return 2
        config_path = run_dir / "config.json"
        if config_path.exists():
            stored = json.loads(config_path.read_text())
            args.catalog = stored.get("catalog", args.catalog)
            args.hardware = stored.get("hardware", args.hardware)
            args.models = stored.get("models", args.models)
            args.precisions = stored.get("precisions", args.precisions)
            if args.comparison_batch == 1000 and "comparison_batch" in stored:
                args.comparison_batch = stored["comparison_batch"]
        logger = build_logger(run_dir)
        catalog_path = Path(args.catalog).resolve()
        hardware_path = Path(args.hardware).resolve()
        try:
            models = measured_models(catalog_path, args.models)
        except FileNotFoundError:
            print(f"ERROR: model catalog not found: {catalog_path}", file=sys.stderr)
            return 2
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR: invalid model catalog {catalog_path}: {exc}", file=sys.stderr)
            return 2
        precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
        problems = preflight(args, catalog_path, hardware_path, models, logger, stage="report")
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 2
        result = pipeline.run_report_stage(run_dir, catalog_path, hardware_path, models, precisions,
                                           args.comparison_batch, not args.no_latex, not args.no_compile_latex,
                                           args.latexmk, logger, texlive_bin=args.texlive_bin)
        if result.get("pdf"):
            print(f"PDF: {result['pdf']}")
        return 0

    run_dir = Path(args.output_dir).expanduser() / args.run_name if args.run_name else (
        Path(args.output_dir).expanduser() / f"run_{datetime.now():%Y%m%d_%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir)
    catalog_path = Path(args.catalog).expanduser().resolve()
    hardware_path = Path(args.hardware).expanduser().resolve()
    try:
        models = measured_models(catalog_path, args.models)
    except FileNotFoundError:
        print(f"ERROR: model catalog not found: {catalog_path}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: invalid model catalog {catalog_path}: {exc}", file=sys.stderr)
        return 2
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    ladder = manual_ladder(args)
    problems = preflight(args, catalog_path, hardware_path, models, logger, stage=args.stage)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 2

    if args.stage:
        if args.stage == "measure":
            precision = args.precision_arg or precisions[0]
            pipeline.run_measure_stage(catalog_path, hardware_path, models, precision, args.memory_fraction,
                                       ladder_step(args.ladder), args.max_batch, ladder, args.warmup, args.iterations,
                                       args.device, run_dir, logger)
            return 0
        if args.stage == "ncu":
            from ncu import DEFAULT_METRICS

            _, ok = pipeline.run_ncu_stage(catalog_path, models, precisions, run_dir, Path(args.ncu_bin),
                                           DEFAULT_METRICS, args.ncu_timeout, logger, args.allow_missing_ncu,
                                           batches_spec=args.ncu_batches, max_batch=args.ncu_max_batch,
                                           cache_control=args.ncu_cache_control, replay_mode=args.ncu_replay_mode,
                                           clock_control=args.ncu_clock_control)
            return 0 if ok else 2
        if args.stage == "nsys":
            from catalog import load_catalog

            pipeline.run_nsys_stage(catalog_path, load_catalog(catalog_path), models, precisions, run_dir,
                                    args.nsys_bin, args.nsys_repeats, args.nsys_timeout, logger)
            return 0
        pipeline.run_report_stage(run_dir, catalog_path, hardware_path, models, precisions, args.comparison_batch,
                                  not args.no_latex, not args.no_compile_latex, args.latexmk, logger,
                                  texlive_bin=args.texlive_bin)
        return 0

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Models: {models} | Precisions: {precisions} | Memory gate: {args.memory_fraction:.0%} | "
                f"Ladder: {args.ladder} | Max batch: {args.max_batch or 'none'} | Comparison batch: {args.comparison_batch}")
    (run_dir / "config.json").write_text(json.dumps(config_for(args), indent=2))
    if args.dry_run:
        logger.info("Dry run: no stages executed")
        return 0

    failures: list[str] = []
    base = [sys.executable, str(ROOT / "analyze.py"), "--catalog", str(catalog_path), "--hardware", str(hardware_path),
            "--models", ",".join(models), "--precisions", ",".join(precisions), "--memory-fraction", str(args.memory_fraction),
            "--ladder", args.ladder, "--comparison-batch", str(args.comparison_batch),
            "--warmup", str(args.warmup), "--iterations", str(args.iterations), "--device", args.device,
            "--output-dir", str(run_dir.parent), "--run-name", run_dir.name,
            "--ncu-bin", args.ncu_bin, "--ncu-timeout", str(args.ncu_timeout), "--nsys-bin", args.nsys_bin,
            "--nsys-repeats", str(args.nsys_repeats), "--nsys-timeout", str(args.nsys_timeout),
            "--ncu-batches", args.ncu_batches, "--ncu-cache-control", args.ncu_cache_control,
            "--ncu-replay-mode", args.ncu_replay_mode, "--ncu-clock-control", args.ncu_clock_control]
    if args.ncu_max_batch is not None:
        base += ["--ncu-max-batch", str(args.ncu_max_batch)]
    if args.texlive_bin:
        base += ["--texlive-bin", args.texlive_bin]
    if args.max_batch is not None:
        base += ["--max-batch", str(args.max_batch)]
    if args.batch_ladder:
        base += ["--batch-ladder", args.batch_ladder]

    for precision in precisions:
        command = [*base, "--stage", "measure", "--precision-arg", precision]
        logger.info("$ " + " ".join(command))
        if subprocess.run(command).returncode != 0:
            failures.append(f"measure/{precision}")
    if not args.skip_ncu:
        command = [*base, "--stage", "ncu"] + (["--allow-missing-ncu"] if args.allow_missing_ncu else [])
        logger.info("$ " + " ".join(command))
        if subprocess.run(command).returncode != 0:
            failures.append("ncu")
    if not args.skip_nsys:
        command = [*base, "--stage", "nsys"]
        logger.info("$ " + " ".join(command))
        if subprocess.run(command).returncode != 0:
            failures.append("nsys")
    command = [*base, "--stage", "report"] + (["--no-latex"] if args.no_latex else []) + (["--no-compile-latex"] if args.no_compile_latex else [])
    if args.latexmk:
        command += ["--latexmk", args.latexmk]
    logger.info("$ " + " ".join(command))
    if subprocess.run(command).returncode != 0:
        failures.append("report")

    if failures:
        logger.error("PIPELINE FAILED: " + ", ".join(failures))
        return 2
    logger.info("PIPELINE PASSED")
    logger.info(f"Artifacts in {run_dir}: tables/, figures/, report/, *summary.json, measurement_*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
