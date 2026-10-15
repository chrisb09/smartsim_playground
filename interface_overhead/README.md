# Co-located benchmark drafts

These are local review figures, not yet integrated into the thesis.

Run from this directory:

```sh
python -B plot_colocated_campaign.py
python -B -m unittest test_plot_colocated_campaign.py
```

The generator uses the sibling `mini_app/summarize_colocated_campaign.py` and
`~/Master-Thesis/gitlab/sourcecode/thesis_style.py`. Matplotlib, pdflatex and a PDF
rasterizer supported by `thesis_style` are required. Paths can be changed with
`--campaign-root`, `--output`, and `--style-path`.

## Outputs

- `figures/interface_scorep_cpu.{png,pdf,pgf}` and
  `figures/interface_scorep_gpu.{png,pdf,pgf}`: Direct without preload, CMI,
  CMI profiling and CMI tracing. Separate 0.49-textwidth, 1.8-inch-high figures
  with independent zero-based linear axes and their own y-labels, for placement
  beside one another with LaTeX subcaptions. No internal titles or subtitle.
- `figures/interface_scorep_time.{png,pdf,pgf}`: retained original two-panel
  draft; no longer regenerated. Preload drafts remain available for review,
  but are not planned for thesis inclusion. No thesis files are modified.
- `figures/preload_time.{png,pdf,pgf}`: Direct without/with terrain preload.
- `figures/preload_payload.{png,pdf,pgf}`: input, output and one-time terrain
  preload, with two Direct bars per category.
- `figures/absolute_time.csv`: absolute means and 95% confidence intervals.
- `figures/job_means.csv`: all 50 condition/job means and source log paths.
- `figures/payload.csv`: Direct payload counters for all 20 condition/job runs.
- `figures/paired_comparisons.csv`: within-job comparisons from the campaign
  analyzer (its Student-t critical value is rounded to 2.776).
- `figures/measurement_summary.json`: the complete exported measurements.

## Measurement design

Campaign: `mini_app/logs/colocated_campaign/colocated_full_4961357`.

Five independent exclusive-node allocations per platform each ran all five
conditions sequentially, with fresh DB/state/output per condition. The condition
order rotates across replicates. Solver and DB share the node's 96 CPUs through
overlapping Slurm steps. The solver uses 96 MPI ranks, a 12x8 decomposition and
one thread per rank. Direct/CMI/Score-P binaries came from the matched Release
build job 4961357; profiling and tracing use the same instrumented binary.

CPU: account rwth2150, partition c23ms, grid 384x216,
TPQ=6/intra=16/inter=1. GPU: account rwth0792, partition c23g, grid 1920x1080,
four inference GPUs, TPQ=1/intra=8/inter=1. These are different workloads:
the panels do not measure CPU/GPU speedup.

Production jobs:

- CPU: 4966611, 4966613, 4966614, 4966618, 4966621.
- GPU: 4966610, 4966612, 4966616, 4966617, 4966620.

Each condition has 66 solver steps with ML every second step: 33 ML calls.
The first three ML calls are discarded; the remaining 30 are averaged within
each job. Column heights are the mean of the five job means, in seconds.
Error bars are two-sided Student-t 95% confidence intervals with four degrees
of freedom (critical value 2.7764451052). Calls are not pooled as independent
replicates. Overlapping absolute-mean CIs are not a test of paired overhead.
Use the paired-comparison export for statements about treatment differences.

The analyzed metric is `STEP_TIMING ... solver=ML step_ms=...`. Production
ML records contain `rank_min_ms` and `rank_max_ms`, establishing timing mode 1
in `terrain_solver.cpp`. The code starts the timer after a start barrier,
includes ML solver work, rank-time reductions and an end barrier, then reports
elapsed time on rank zero. It excludes preceding halo exchange and subsequent
state swap, reporting and snapshot I/O. This is synchronized ML-step time,
not isolated inference latency or total application wall time.

Payload columns show logged logical MiB summed over ranks for all 33 calls,
including warmup, and the one-time terrain upload separately. Counters agree
across all five jobs, so no payload error bars are drawn. Printed counters have
limited precision. These are not observed network bytes. CMI traffic counters
are unavailable (reported as zero) and are deliberately omitted.

## Limitations

Successful execution and complete timings do not establish numerical validity.
Final printed scalar diagnostics agree across conditions, but the workload has
negative water mass/heights; fieldwise equivalence and the source of this
nonphysical behavior remain unverified. Performance results are provisional
controlled-workload measurements, not validated physical simulation results.

All runs use CUDA_LAUNCH_BLOCKING=1 and disabled PAPI metrics. GPU-kernel and
Redis-internal attribution cannot be inferred from the solver traces alone.
Score-P warmup region labeling does not match the three-call exclusion used
here; reconcile that before using profile/trace aggregates as timing evidence.

The generator checks all five jobs/conditions, exact step IDs and parity,
three-call exclusion, 30 retained calls, synchronized ML records and identical
Direct payloads across jobs. It does not independently audit binary hashes,
every rank receipt, trajectory equivalence or Score-P artifact contents.
