# Measured Model Inspection and Roofline Analysis

Measures the inference cost of TorchScript models along four axes and produces
tables, figures, and a LaTeX report. All reported values are measurements; no
analytical memory model is used.

- **Arithmetic**: executed hardware instructions from Nsight Compute counters
  (`2*FFMA + FADD + FMUL + tensor-core ops`; tensor counters already count
  one multiply and one add per MAC).
- **HBM3 RAM / L2**: NCU device-memory (DRAM) read/write counters and L2 request/sector counters,
  collected with warm caches (the model has run before profiling).
- **PCIe**: Nsight Systems CUPTI memcpy payload per inference.
- **Latency / throughput**: CUDA-event timing of H2D, forward, D2H, and the
  full step, plus adaptive VRAM scaling.

## Layout

```text
analyze.py                    public CLI (argparse; see --help)
measure_stats.py              repeated timing/counter statistics at B=1 and ~80% VRAM
run_analysis.sbatch           Slurm wrapper (submit --exclusive for campaigns)
run_stats.sbatch              Slurm wrapper for the repeated-statistics script
hardware_h100.json            HBM/compute ceilings; link values get recalibrated
model_catalog.json            artifact paths + input/output contracts
set_env_claix23_cuda12.4.sh   local copy of the cluster module environment (sourced by the wrapper)
src/                          implementation modules and LaTeX template
artifacts/test_mlp/           test MLP artifacts (only needed by the canonical catalog)
generated/<run>/              one self-contained run directory per pipeline run
.old/                         archived legacy scripts, exploration, old results
```

Each run directory contains:

- `measurement_<precision>.json` — adaptive VRAM discovery, storage, workspace,
  CUDA-event timings, link calibration, GPU identity.
- `ncu_summary.json` + `raw/ncu_*.csv` — RAM (DRAM counters), L2, and hardware arithmetic
  counters with cache/replay mode metadata.
- `nsys_summary.json` + `raw/nsys_*.{nsys-rep,sqlite}` — per-config memcpy records.
- `tables/*.csv` — all result tables (see below).
- `figures/*.{pdf,png}` — measured bar charts, RAM/L2 figures, measured
  roofline, and throughput.
- `report/report.tex`, `report/report.pdf` — generated LaTeX report.
- `provenance.json`, `config.json`, `pipeline.log`.

## Quick Start

The project also runs without any trained artifacts by using the untrained
replicas: pass `--catalog model_catalog_replicas.json` (missing replicas are
generated on demand, see Replicas below). The default `model_catalog.json`
points at the original artifacts and only works where those exist.

Login-node smoke run on a free GPU (pin by UUID; check `nvidia-smi -L` and
`--query-compute-apps` first, several login GPUs run in `Exclusive_Process`
mode and may be occupied by other users):

```bash
CUDA_VISIBLE_DEVICES=GPU-8ff8d0c7-8d30-8e55-0980-ac69fc03a6b8 \
python analyze.py --catalog model_catalog_replicas.json --models mmcp_test_mlp_m5 --run-name smoke
```

Profile every discovered ladder batch with NCU (warm caches, application replay):

```bash
python analyze.py --catalog model_catalog_replicas.json --models watercnn --ncu-batches ladder --run-name ladder_test
```

Full exclusive campaign (NCU over each model's full discovered ladder):

```bash
CATALOG=model_catalog_replicas.json sbatch --exclusive run_analysis.sbatch
```

Regenerate tables/figures/report from an existing run (CPU only):

```bash
python analyze.py --from-run generated/<run>
```

Environment overrides for the Slurm wrapper: `CATALOG` (default `model_catalog.json`), `MODELS`, `PRECISIONS` (default
`fp32`), `MEMORY_FRACTION`, `LADDER` (x<N>, wrapper default `x8`), `MAX_BATCH`, `BATCH_LADDER`,
`COMPARISON_BATCH` (default 1000), `NCU_BIN`, `NCU_BATCHES` (default `ladder`;
use e.g. `1,1000` to cap), `NCU_MAX_BATCH`, `NCU_CACHE_CONTROL` (default
`none`), `NCU_REPLAY_MODE` (default `application`), `NCU_CLOCK_CONTROL`
(default `none`), `NSYS_BIN`, `SKIP_NCU=1`, `SKIP_NSYS=1`, `NO_LATEX=1`,
`NO_COMPILE_LATEX=1`, `LATEXMK` (path to the latexmk binary), `TEXLIVE_BIN` (TeX bin directory prepended to PATH when compiling), `EXTRA_ARGS`.

## Repeated Measurement Statistics

`measure_stats.py` runs the CUDA-event benchmark, the NCU counter pass, and an
Nsight Systems PCIe trace per batch, each `--repeats` times (default 10), at two
batches per model: B=1 and a large batch whose predicted peak VRAM is close to
`--memory-fraction` (default 0.8) of device memory. The large batch is derived
from a stored VRAM fit (newest matching `generated/*/measurement_<precision>.json`,
or `--fit-from-run`) and probed fresh when no fit is available (`--probe` forces
probing). The fit target is rounded down to two significant digits (113 M ->
110 M), verified with the timing workload, stepped down by one significant digit
when it fails, and then bisected upward between the passing and failing value so
kernel-limited models land on their feasible maximum (Transformer -> 65 000).
Every timing phase, VRAM peak, raw NCU counter, derived FLOP/OI value, and
per-inference PCIe payload is reported as `n`, median, mean, sample std, min,
max, p25, p75, and CV in `generated/<run>/stats.csv` plus the full `stats.json`.

```bash
python measure_stats.py --skip-ncu --repeats 3 --models mmcp_test_mlp_m5 --run-name stats_smoke
python measure_stats.py --repeats 10 --run-name stats_campaign
CATALOG=model_catalog_replicas.json sbatch --exclusive run_stats.sbatch
```

`run_stats.sbatch` defaults to `REPEATS=5`; the script itself defaults to 10.
The NCU pass dominates the runtime, so `--skip-ncu` gives a quick timing-only
comparison, `--skip-nsys` drops the PCIe traces, and `--max-batch` caps the
large batch for tests. Flags passed after the script name are forwarded to
`measure_stats.py` and override the environment defaults (CLAIX does not export
submit-host variables into the batch environment):

```bash
sbatch --exclusive run_stats.sbatch --models tbl_transformer --warmup 5 --iterations 10
```

`--resume` continues an existing run directory, skipping every batch that
already has the requested number of timing, NCU, and PCIe repetitions, so a
timed-out job can be topped up model by model. `--rebatch <models>` additionally
re-selects those models' large batch while keeping completed batches (used to
correct the adopted B_max after changing the rounding rule):

```bash
CATALOG=model_catalog.json sbatch --exclusive run_stats.sbatch \
    --run-name slurm_stats_<oldjob> --resume --rebatch watercnn,giant_mlp \
    --warmup 5 --iterations 10
```

## Adaptive Batch Discovery

Each model gets its own batch ladder instead of a fixed list. Starting from
`B=1`, the next candidate is the current largest times the ladder multiplier
`--ladder x<N>` (default `x10`, e.g. `x2`, `x5`; `N` must be at least 2).
Every point runs one forward and records peak allocated and reserved VRAM; a
two-point line through `B=1` and the current largest predicts the candidate's
memory, and growth stops once the prediction exceeds `--memory-fraction`
(default 0.9) of total device memory, an actual OOM occurs, or `--max-batch`
is hit. `--batch-ladder 1,10,100` bypasses discovery. The internal fit is a
safety check only and is not reported; timing runs over the full ladder.

## Replicas

`model_catalog_replicas.json` mirrors the canonical catalog but points at
untrained, contract-compatible TorchScript replicas. Missing artifacts are
generated on demand by `src/build_model_replicas.py` into `{replica_root}`
(default `/tmp/$USER/model_inspection_replicas`, override with
`MODEL_REPLICA_ROOT`). Use it directly (`--catalog model_catalog_replicas.json`)
or via the Slurm wrapper (`CATALOG=model_catalog_replicas.json`).

Replicas are for workflow and performance-method validation, not scientific
results. Login-node validation shows parameter counts, FLOPs, and L2 traffic
match the real artifacts for the CNN and MLP models (the TBL Transformer
replica mirrors the real autoregressive two-step execution); small-batch DRAM
residency and absolute timings can still differ, and random weights mean the
numerical outputs are not the trained models'.

## NCU Batch Selection

`--ncu-batches` controls which batches Nsight Compute profiles:

- `1,1000` (the CLI default): explicit batch sizes, each snapped to the nearest
  batch of that model's discovered ladder (for example `1000 -> 1024` on an
  `x2` ladder), so NCU, timing, and tables share one batch grid.
- `ladder` / `full`: every batch of each model's discovered ladder (the Slurm
  wrapper uses this).
- `--ncu-max-batch N` drops ladder points above `N`.

`run_analysis.sbatch` therefore profiles full ladders by default; use
`NCU_BATCHES=1,1000` to fall back to two points per model. The comparison batch
in the report (`--comparison-batch`, default 1000) is likewise resolved to the
closest batch that exists for all models in both timing and NCU data.

## Measurement Passes

Each forward is re-executed per pass because NCU requires its own process
(exclusive CUPTI counter access; Nsight Systems injection would block it) and
because per-pass isolation keeps replay effects out of the timings.

| Pass | Scope | What it produces | Mechanism |
|---|---|---|---|
| 1. Measurement | per-model adaptive ladder | VRAM scaling points, storage, workspace, H2D/forward/D2H/step timings and samples/s, link calibration | CUDA events + peak-memory stats |
| 2. NCU | `--ncu-batches` (default B=1/B=1000, Slurm: full ladder) | per-kernel and aggregate RAM (DRAM counter) read/write, L2 bytes and read/write sectors, SASS FFMA/FADD/FMUL, per-precision tensor-core ops (TF32/FP16/BF16/FP8/INT8/FP64) | NCU >= 2025.1 (CUDA 12.8 binary; 2024.1 fails on driver 580.x), `--cache-control none`, `--replay-mode application` |
| 3. Nsight Systems | B=1 and B=1000 per model/precision | exact per-inference H2D/D2H memcpy payload (NVTX `nsys_audit` window) | `nsys profile -t cuda,nvtx` + SQLite CUPTI activity records |

Cache and replay semantics matter and are recorded in `ncu_summary.json`:

- `--cache-control none` measures warm caches: the model's code, constants, and
  weights stay in the 50 MB L2 between passes, so small-batch RAM traffic
  reflects a model that has already run rather than a cold-start microbenchmark.
- `--replay-mode application` re-runs the whole process once per metric pass.
  This is much faster than NCU's default per-kernel replay for models with many
  kernels or large memory footprints. `range` replay is not usable with the
  SASS instruction counters; `kernel` remains available as a fallback.
- RAM writes are a lower bound: write-back caches can retain outputs during
  the measurement window.

Counter-derived FLOPs are executed hardware work
(`2*FFMA + FADD + FMUL + tensor-core ops`), reported next to the raw counters in
`flops_measured_hardware.csv`; they can exceed the mathematical operation count
when the backend chooses a different algorithm (implicit-GEMM or Winograd
convolutions). The tensor `ops_path` counters count one multiply and one add
per MAC — i.e. they are already FLOPs, verified against a 1024x1024x1024 GEMM
(counter = 2^31 for 2^30 MACs) — and therefore enter the sum unweighted, while
FFMA covers both operations of the FMA pipe.

## CSV Tables (`generated/<run>/tables/`)

Long-form tables use `model_id, display_name, precision, batch_size, metric,
value, unit, source, note`. Missing values are empty, never zero.

| Table | Contents |
|---|---|
| `model_storage.csv` | parameter/buffer counts and bytes by dtype (inspection) |
| `flops_measured_hardware.csv` | raw NCU arithmetic counters and `measured_flops` |
| `precision_ops_measured.csv` | executed FLOPs split by precision (FP32 FMA pipe, TF32/FP16/BF16/FP8/INT8/FP64 tensor cores) |
| `hbm3_measured.csv` | NCU RAM read/write/total bytes (DRAM counters) |
| `cache_l2_measured.csv` | RAM versus L2 requests plus the L2/RAM ratio |
| `pcie_measured.csv` | Nsys memcpy payload per inference |
| `oi_hbm3_measured.csv` | measured operational intensity (FLOP / RAM byte) |
| `timing_measured.csv` | phase latencies (ms) and samples/s |

## Figures

- `bar_flops`, `bar_hbm3`, `bar_pcie`: measured values at the comparison batch
  (default B=1000), log y-axis, numeric labels.
- `bar_precision`: stacked 100% bars with the share of executed FLOPs per
  precision (FP32 FMA pipe versus tensor-core precisions).
- `bar_l2_vs_dram`: L2 request bytes vs RAM bytes with the L2/RAM ratio.
- `roofline_hbm3_measured`: counter-derived FLOPs per measured DRAM byte vs
  counter-derived FLOP/s, with FP32 and (when tensor work is present) tensor-core
  (TF32) ceilings as reference lines.
- `hbm_vs_batch`: DRAM read and write traffic versus batch size.
- `oi_vs_batch`: measured operational intensity versus batch size; the decline
  marks the transition from L2-resident to DRAM-bound execution.
- `throughput_<precision>`: serial end-to-end samples/s over batch size.

## LaTeX Report

`analyze.py` renders `src/templates/report.tex.tpl` into
`generated/<run>/report/report.tex` and, when `latexmk` is available, compiles
`report.pdf`. The report contains the objective, the model table with long
names and group explanations (TS / TBL), measurement methodology subdivided by
axis (arithmetic, memory, transfer, timing, adaptive discovery), measured
tables at B=1 and the comparison batch, all figures, and limitations. Table
numbers are digit-aligned with thin-space grouping and captions sit below the
tables. It compiles with TeX Live 2026
(`~/opt/texlive/2026/bin/x86_64-linux/latexmk`); the renderer prepends the
latexmk directory (or an explicitly configured `--texlive-bin`) to `PATH` so an
older system TeX cannot shadow it.

## Notes

- `build_model_replicas.py` (in `src/`) emits untrained contract-compatible
  replicas plus a canonical `model_catalog.json`, so the pipeline can run
  against `--catalog <replicas>/model_catalog.json`.
- Catalog naming: every model has `short_name` (used in tables and figures),
  `long_name` (introduced once per report), and a short `notes` field that is
  usually empty (the TBL Transformer notes its partial tensor-core (TF32) use).
  `groups` explains the `TS` (Terrain Solver, fixed 18-to-1 patch contract) and
  `TBL` (Turbulent Boundary Layer, sequence length fixed to 5) families, and
  `uses_tf32` marks models that execute tensor-core arithmetic.
- Legacy investigation notes live in `.old/exploration/ncu_smoke/SUMMARY.md`
  (NCU 2024.1 root cause, Nsys/NCU incompatibility evidence, preliminary
  campaign numbers). The former flat layout (measurement scripts, old
  orchestrator, preflight jobs) is archived under `.old/scripts/`.
- The earlier analytical traffic-model work (unique-bytes lower bound and
  per-operator operand model) was removed from the pipeline; the findings
  remain in git history and the exploration notes.
