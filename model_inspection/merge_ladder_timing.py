#!/usr/bin/env python3
"""Merge the per-invocation measure-stage runs of the x2 ladder-timing campaign
into a single long-form CSV that matches the ``timing_measured`` table schema.

The campaign runs one ``analyze.py`` invocation per ladder (a manual batch
ladder is global to an invocation), and the memory-hungry MLP ladder is split
across two invocations (``mlp_a``, ``mlp_b``) because PyTorch's caching host
allocator accumulates pinned I/O buffers within a process. This script scans
every run directory under ``--run-root`` and groups the per-batch CUDA-event
timings by model, so split ladders are reassembled automatically.

Usage:
  python3 merge_ladder_timing.py --run-root generated/slurm_4491052 \
      --output data/model_inspection/ladder_timing_4491052.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HEADER = ["model_id", "display_name", "precision", "batch_size", "metric", "value", "unit", "source", "note"]

# Thesis labels (the catalog short names lack the [TS]/[TBL] group prefix).
LABELS = {
    "watercnn": "[TS] WaterCNN",
    "mmcp_test_mlp_m5": "[TBL] MLP",
    "tbl_transformer": "[TBL] Transformer",
    "giant_mlp": "[TS] Giant MLP",
}
ORDER = ["watercnn", "mmcp_test_mlp_m5", "tbl_transformer", "giant_mlp"]

METRICS = (
    ("h2d_ms", "median_h2d_ms", "ms"),
    ("forward_ms", "median_forward_ms", "ms"),
    ("d2h_ms", "median_d2h_ms", "ms"),
    ("step_ms", "median_step_ms", "ms"),
    ("resident_forward_ms", "median_resident_forward_ms", "ms"),
    ("resident_samples_per_s", "resident_samples_per_s", "samples/s"),
    ("serial_samples_per_s", "serial_samples_per_s", "samples/s"),
)
NOTE = "x2 ladder, single run per batch; median of 30 timed iterations after 10 warm-up forwards"


def collect(run_root: Path, precision: str) -> dict[str, dict[int, dict]]:
    """Return {model_id: {batch: timing}} merged across all run directories."""
    per_model: dict[str, dict[int, dict]] = {model_id: {} for model_id in ORDER}
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        path = run_dir / f"measurement_{precision}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for record in data["models"]:
            model_id = record["model_id"]
            if model_id not in per_model:
                continue
            for timing in record["batches"]:
                per_model[model_id][int(timing["batch_size"])] = timing
    return per_model


def merge(run_root: Path, output: Path, precision: str = "fp32") -> list[list]:
    per_model = collect(run_root, precision)
    rows: list[list] = []
    for model_id in ORDER:
        label = LABELS.get(model_id, model_id)
        for batch in sorted(per_model[model_id]):
            timing = dict(per_model[model_id][batch])
            resident_ms = timing.get("median_resident_forward_ms")
            step_ms = timing.get("median_step_ms")
            timing.setdefault("resident_samples_per_s", batch / (resident_ms / 1000) if resident_ms else None)
            timing.setdefault("serial_samples_per_s", batch / (step_ms / 1000) if step_ms else None)
            for metric, key, unit in METRICS:
                value = timing.get(key)
                rows.append([model_id, label, precision, batch, metric,
                             "" if value is None else value, unit, "cuda_events", NOTE])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ladder-timing runs into one CSV.")
    parser.add_argument("--run-root", required=True, type=Path, help="e.g. generated/slurm_4491052")
    parser.add_argument("--output", required=True, type=Path, help="output CSV path")
    parser.add_argument("--precision", default="fp32")
    args = parser.parse_args()

    rows = merge(args.run_root, args.output, args.precision)
    print(f"wrote {len(rows)} rows to {args.output}")
    for model_id in ORDER:
        serial = [(r[3], float(r[5])) for r in rows if r[0] == model_id and r[4] == "serial_samples_per_s"]
        if serial:
            print(f"  {LABELS.get(model_id, model_id)}: {len(serial)} points, "
                  f"B={serial[0][0]} -> {serial[0][1]:,.0f} samples/s, "
                  f"B={serial[-1][0]} -> {serial[-1][1]:,.0f} samples/s")


if __name__ == "__main__":
    main()
