#!/usr/bin/env python3
"""
model_inspection/plot_roofline.py
==================================
Roofline plot for all models in an inspection JSON result file.

Produces two figures:
  1. Coupling roofline  — x = coupling OI (FLOPs / I/O bytes per entry),
                           y = achieved GFLOP/s at each batch size.
     Interpretation: where does the model sit relative to the MPI/TCP link?
  2. Device roofline    — x = device OI (FLOPs / (I/O + weights/B) bytes),
                           y = achieved GFLOP/s at each batch size.
     Interpretation: where does the model sit on the GPU memory hierarchy?

Both figures share the same roofline ceiling (H100 NVL 94 GB, CLAIX-23):
  - HBM3 bandwidth slope:  3.938 TB/s
  - FP32 non-TC peak:     60.32 TFLOP/s  (ridge ~15 FLOP/B)
  - TF32 TC dense peak:  482.6  TFLOP/s  (ridge ~123 FLOP/B)

OI is on the x-axis (log scale). Performance is on the y-axis (log scale,
GFLOP/s). The roofline is the minimum of the bandwidth line and the peak.

When timing data is present, each model is plotted as a set of dots (one per
batch size) connected by a trajectory line, showing how the achieved GFLOP/s
and device OI change with batch size. When only static data is available (e.g.
--skip-timing was used), models appear as vertical markers on the x-axis only
(OI known, performance unknown).

Usage:
  python plot_roofline.py results/inspection_<timestamp>_cuda.json
  python plot_roofline.py results/inspection_<timestamp>_cuda.json \\
      --output-dir results/ --format pdf,png --no-device

  # Overlay multiple JSON files (e.g. merge static + timed runs):
  python plot_roofline.py results/A.json results/B.json --output-dir results/

Options:
  --output-dir DIR    Where to write output files (default: same dir as first JSON)
  --format FMTS       Comma-separated output formats: pdf,png,svg (default: pdf,png)
  --no-coupling       Skip coupling roofline figure
  --no-device         Skip device roofline figure
  --batch-sizes BSS   Comma-separated subset of batch sizes to plot (default: all)
  --title-suffix STR  Append to figure title (e.g. hardware name)
  --thesis            Use thesis-friendly style (larger fonts, no interactive elements)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on compute nodes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants — H100 NVL 94 GB, CLAIX-23 c23g partition
# ---------------------------------------------------------------------------
H100_HBM_BW_TFLOPS   = 3.938          # TB/s (= TFLOP/B at 1 FLOP/op)
H100_FP32_PEAK_GFLOPS = 60_320.0      # GFLOP/s, non-TC FP32
H100_TF32_PEAK_GFLOPS = 482_600.0     # GFLOP/s, TF32 dense TC
H100_HBM_BW_GBPS      = 3_938.0       # GB/s (used for roofline slope)

RIDGE_FP32  = H100_FP32_PEAK_GFLOPS / H100_HBM_BW_GBPS   # ~15.3 FLOP/B
RIDGE_TF32  = H100_TF32_PEAK_GFLOPS / H100_HBM_BW_GBPS   # ~122.7 FLOP/B

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011)
_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # pink
    "#000000",  # black
]

# Marker cycle per batch size (up to 6 batch sizes)
_MARKERS = ["o", "s", "^", "D", "v", "P"]

# Model families → line style
_FAMILY_LS = {
    "local_surrogate": "-",
    "mmcp":            "--",
    "benchmark":       ":",
}


def _roofline_ceiling(oi_arr: np.ndarray, peak_gflops: float, bw_gbps: float) -> np.ndarray:
    """Roofline ceiling = min(peak, bw * OI) in GFLOP/s, for an array of OI values."""
    return np.minimum(peak_gflops, bw_gbps * oi_arr)


def _build_roofline_x(x_min: float, x_max: float, n: int = 500) -> np.ndarray:
    return np.logspace(np.log10(x_min), np.log10(x_max), n)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(json_paths: list[Path]) -> tuple[dict, list[dict]]:
    """
    Load one or more inspection JSON files.  When multiple files are given,
    model entries are merged by model_id: later files overwrite earlier ones
    for the same id.  run_meta is taken from the first file.
    """
    run_meta: dict = {}
    by_id: dict[str, dict] = {}
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        if not run_meta:
            run_meta = data.get("run_meta", {})
        for r in data.get("results", []):
            mid = r.get("model_id", "unknown")
            if mid in by_id:
                # Merge: update timing/memory from later file if present
                existing = by_id[mid]
                if r.get("timing") and not r["timing"].get("skipped"):
                    existing["timing"] = r["timing"]
                if r.get("memory", {}).get("activation_by_batch"):
                    existing.setdefault("memory", {})["activation_by_batch"] = \
                        r["memory"]["activation_by_batch"]
            else:
                by_id[mid] = r
    return run_meta, list(by_id.values())


# ---------------------------------------------------------------------------
# Extract plottable points
# ---------------------------------------------------------------------------

def extract_points(results: list[dict], batch_sizes_filter: Optional[list[int]] = None
                   ) -> list[dict]:
    """
    For each model, extract a list of (oi, achieved_gflops, batch_size) triples
    from timing data, plus the coupling_oi (scalar).

    Returns a list of model dicts:
      {
        "model_id": str,
        "display_name": str,
        "family": str,
        "coupling_oi": float | None,         # static, batch-independent
        "flops_per_entry": int | None,
        "points": [                           # one per batch size with timing
            {"batch_size": int, "device_oi": float, "achieved_gflops": float | None}
        ],
        "static_device_oi": {batch_size: float},  # from metrics.device_oi_by_batch
        "has_timing": bool,
      }
    """
    out = []
    for r in results:
        metrics  = r.get("metrics", {})
        timing   = r.get("timing", {})
        flops_pe = metrics.get("flops_per_entry")
        coup_oi  = metrics.get("coupling_oi_flop_per_byte")
        dev_oi_by_batch = metrics.get("device_oi_by_batch", {})

        has_timing = bool(timing) and not timing.get("skipped")

        points = []
        # Batch sizes come from the timing dict keys (int or str) or from device_oi_by_batch
        candidate_bs = set()
        if has_timing:
            candidate_bs |= {int(k) for k in timing.keys() if str(k).isdigit()}
        candidate_bs |= {int(k) for k in dev_oi_by_batch.keys()}
        if batch_sizes_filter:
            candidate_bs = candidate_bs & set(batch_sizes_filter)

        for bs in sorted(candidate_bs):
            dev_oi = dev_oi_by_batch.get(bs) or dev_oi_by_batch.get(str(bs))
            achieved_gflops = None
            if has_timing:
                t = timing.get(bs) or timing.get(str(bs), {})
                achieved_gflops = t.get("achieved_gflops_per_s")
            points.append({
                "batch_size":      bs,
                "device_oi":       dev_oi,
                "achieved_gflops": achieved_gflops,
            })

        out.append({
            "model_id":         r.get("model_id", "unknown"),
            "display_name":     r.get("display_name", r.get("model_id", "?")),
            "family":           r.get("family", "unknown"),
            "coupling_oi":      coup_oi,
            "flops_per_entry":  flops_pe,
            "points":           points,
            "static_device_oi": {int(k): v for k, v in dev_oi_by_batch.items()},
            "has_timing":       has_timing,
        })
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _draw_roofline_ceilings(ax: plt.Axes, x_arr: np.ndarray,
                             label_ridges: bool = True) -> None:
    """Draw HBM bandwidth slope and FP32 / TF32 peaks onto ax."""
    # Bandwidth-limited slope
    bw_line = H100_HBM_BW_GBPS * x_arr
    ax.plot(x_arr, bw_line, color="grey", lw=1.2, ls="-", zorder=1,
            label=f"HBM3 BW ({H100_HBM_BW_GBPS:.0f} GB/s)")

    # FP32 non-TC ceiling
    fp32_ceil = _roofline_ceiling(x_arr, H100_FP32_PEAK_GFLOPS, H100_HBM_BW_GBPS)
    ax.plot(x_arr, fp32_ceil, color="steelblue", lw=1.5, ls="--", zorder=1,
            label=f"FP32 non-TC ({H100_FP32_PEAK_GFLOPS/1e3:.1f} TFLOP/s)")

    # TF32 TC ceiling
    tf32_ceil = _roofline_ceiling(x_arr, H100_TF32_PEAK_GFLOPS, H100_HBM_BW_GBPS)
    ax.plot(x_arr, tf32_ceil, color="tomato", lw=1.5, ls="--", zorder=1,
            label=f"TF32 TC ({H100_TF32_PEAK_GFLOPS/1e3:.0f} TFLOP/s)")

    if label_ridges:
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else H100_TF32_PEAK_GFLOPS
        ax.axvline(RIDGE_FP32, color="steelblue", lw=0.8, ls=":", alpha=0.6)
        ax.axvline(RIDGE_TF32, color="tomato",    lw=0.8, ls=":", alpha=0.6)
        ax.text(RIDGE_FP32 * 1.05, ymax * 0.6, f"FP32\nridge\n~{RIDGE_FP32:.0f}",
                color="steelblue", fontsize=7, va="top")
        ax.text(RIDGE_TF32 * 1.05, ymax * 0.6, f"TF32\nridge\n~{RIDGE_TF32:.0f}",
                color="tomato",    fontsize=7, va="top")


def _oi_axis_label(kind: str) -> str:
    if kind == "coupling":
        return "Coupling Operational Intensity  [FLOP / (in+out) byte]"
    return "Device Operational Intensity  [FLOP / (in+out+weights/B) byte]"


def _plot_one_roofline(
    ax: plt.Axes,
    model_points: list[dict],
    kind: str,                # "coupling" | "device"
    batch_sizes_shown: list[int],
    thesis_style: bool,
) -> None:
    """
    Draw one roofline panel (coupling or device) onto ax.

    For coupling: each model is a single vertical line / dot at coupling_oi.
    For device:   each model has one dot per batch size, connected by a line.
    """
    # Collect all OI values to set axis limits
    all_oi: list[float] = []
    all_gf: list[float] = []
    for m in model_points:
        if kind == "coupling" and m["coupling_oi"] is not None:
            all_oi.append(m["coupling_oi"])
        elif kind == "device":
            for pt in m["points"]:
                if pt["device_oi"] is not None:
                    all_oi.append(pt["device_oi"])

    if not all_oi:
        ax.text(0.5, 0.5, "No OI data", transform=ax.transAxes, ha="center")
        return

    x_min = max(1e-2, min(all_oi) / 3)
    x_max = max(all_oi) * 4
    x_arr = _build_roofline_x(x_min, x_max)

    _draw_roofline_ceilings(ax, x_arr, label_ridges=True)

    # Determine y-range: start with roofline ceiling, extend if achieved values exist
    y_ceil_max = H100_TF32_PEAK_GFLOPS
    for m in model_points:
        for pt in m["points"]:
            if pt.get("achieved_gflops") is not None:
                all_gf.append(pt["achieved_gflops"])
    y_min = 1e-1
    y_max = y_ceil_max * 3

    # Plot each model
    for idx, m in enumerate(model_points):
        color  = _PALETTE[idx % len(_PALETTE)]
        ls     = _FAMILY_LS.get(m["family"], "-")
        name   = m["display_name"]
        has_t  = m["has_timing"]

        if kind == "coupling":
            oi = m["coupling_oi"]
            if oi is None:
                continue
            if has_t:
                # Plot one dot per batch size at the same coupling_oi x-position,
                # with achieved GFLOP/s as y.
                ys = []
                for pt in m["points"]:
                    gf = pt.get("achieved_gflops")
                    if gf is not None:
                        ys.append(gf)
                if ys:
                    # Vertical spread — jitter x slightly per batch size for readability
                    for j, (pt, gf) in enumerate(
                        [(pt, pt["achieved_gflops"]) for pt in m["points"]
                         if pt.get("achieved_gflops") is not None]
                    ):
                        bs = pt["batch_size"]
                        marker = _MARKERS[batch_sizes_shown.index(bs) % len(_MARKERS)] \
                            if bs in batch_sizes_shown else "o"
                        ax.scatter([oi], [gf], color=color, marker=marker,
                                   s=60, zorder=5, edgecolors="white", linewidths=0.5)
                    # Connect dots with a vertical line at x=oi
                    ax.plot([oi] * len(ys), ys, color=color, lw=0.8, ls=ls,
                            alpha=0.5, zorder=4)
                    # Label at top dot
                    ax.annotate(
                        name, xy=(oi, max(ys)),
                        xytext=(4, 2), textcoords="offset points",
                        fontsize=7 if not thesis_style else 8,
                        color=color, va="bottom",
                    )
                else:
                    # No achieved GFLOP/s — draw a vertical dashed line on x only
                    ax.axvline(oi, color=color, lw=1.0, ls=":", alpha=0.7,
                               label=name)
            else:
                # Static only — draw vertical marker at x=oi, no y data
                ax.axvline(oi, color=color, lw=1.2, ls=":", alpha=0.8)
                ax.text(oi * 1.04, y_min * 2, name,
                        color=color, fontsize=7, rotation=90, va="bottom")

        else:  # device roofline
            xs, ys = [], []
            for pt in m["points"]:
                if pt["batch_size"] not in batch_sizes_shown:
                    continue
                oi_val = pt["device_oi"]
                gf_val = pt.get("achieved_gflops")
                if oi_val is None:
                    continue
                if not has_t or gf_val is None:
                    # No y — plot on the bandwidth slope at x=oi_val as hollow marker
                    y_slope = min(H100_HBM_BW_GBPS * oi_val, H100_TF32_PEAK_GFLOPS)
                    ax.scatter([oi_val], [y_slope], color=color, marker="x",
                               s=50, zorder=4, alpha=0.5)
                else:
                    xs.append(oi_val)
                    ys.append(gf_val)

            if xs:
                ax.plot(xs, ys, color=color, lw=1.2, ls=ls, alpha=0.7, zorder=3)
                for j, (x, y, pt) in enumerate(
                    zip(xs, ys, [p for p in m["points"] if p["batch_size"] in batch_sizes_shown
                                 and p.get("achieved_gflops") is not None])
                ):
                    bs = pt["batch_size"]
                    marker = _MARKERS[batch_sizes_shown.index(bs) % len(_MARKERS)] \
                        if bs in batch_sizes_shown else "o"
                    sc = ax.scatter([x], [y], color=color, marker=marker,
                                    s=65, zorder=5, edgecolors="white", linewidths=0.5,
                                    label=f"{name} B={bs}" if j == 0 else "_")
                # Annotate at the rightmost (largest batch) point
                ax.annotate(
                    name, xy=(xs[-1], ys[-1]),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=7 if not thesis_style else 8,
                    color=color, va="bottom",
                )

    # Batch-size legend handles (shared markers across models)
    bs_handles = []
    for j, bs in enumerate(batch_sizes_shown):
        bs_handles.append(
            plt.Line2D([0], [0], marker=_MARKERS[j % len(_MARKERS)],
                       color="grey", linestyle="None",
                       markersize=6, label=f"B={bs}")
        )

    # Roofline legend (first legend)
    leg1 = ax.legend(loc="upper left", fontsize=7, framealpha=0.85,
                     title="Hardware limits", title_fontsize=7)
    ax.add_artist(leg1)
    # Batch-size legend (second legend)
    if bs_handles and kind == "device":
        ax.legend(handles=bs_handles, loc="lower right", fontsize=7,
                  framealpha=0.85, title="Batch size", title_fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(_oi_axis_label(kind),
                  fontsize=9 if not thesis_style else 11)
    ax.set_ylabel("Achieved Performance  [GFLOP/s]",
                  fontsize=9 if not thesis_style else 11)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    ax.tick_params(axis="both", labelsize=7 if not thesis_style else 9)


# ---------------------------------------------------------------------------
# Top-level figure builders
# ---------------------------------------------------------------------------

def plot_coupling_roofline(
    model_points: list[dict],
    run_meta: dict,
    batch_sizes_shown: list[int],
    thesis_style: bool,
    title_suffix: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5) if not thesis_style else (10, 6))
    device = run_meta.get("cuda_device_name", run_meta.get("device", "?"))
    title = f"Coupling Roofline — {device}"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.set_title(title, fontsize=10 if not thesis_style else 12)

    _plot_one_roofline(ax, model_points, "coupling", batch_sizes_shown, thesis_style)

    note = (
        "x-axis: FLOPs per entry / (input + output bytes per entry).\n"
        "Relevant bottleneck bandwidth: MPI cross-node ~5.6 GiB/s, intra-node ~8.8 GiB/s (CLAIX-23 measured).\n"
        "OI uses weight payload as proxy for HBM traffic (arithmetic intensity roofline)."
    )
    fig.text(0.5, -0.04, note, ha="center", fontsize=6, style="italic",
             color="dimgrey", wrap=True)
    fig.tight_layout()
    return fig


def plot_device_roofline(
    model_points: list[dict],
    run_meta: dict,
    batch_sizes_shown: list[int],
    thesis_style: bool,
    title_suffix: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5) if not thesis_style else (11, 6))
    device = run_meta.get("cuda_device_name", run_meta.get("device", "?"))
    title = f"Device Roofline — {device}"
    if title_suffix:
        title += f"  ({title_suffix})"
    ax.set_title(title, fontsize=10 if not thesis_style else 12)

    _plot_one_roofline(ax, model_points, "device", batch_sizes_shown, thesis_style)

    note = (
        "x-axis: FLOPs per entry / (in+out + weights/B bytes). "
        "Dots move right as B increases (weights amortised). "
        "Roofline: H100 NVL 94 GB, HBM3 3.938 TB/s. "
        "OI uses weight payload as proxy for HBM traffic."
    )
    fig.text(0.5, -0.04, note, ha="center", fontsize=6, style="italic",
             color="dimgrey", wrap=True)
    fig.tight_layout()
    return fig


def plot_combined(
    model_points: list[dict],
    run_meta: dict,
    batch_sizes_shown: list[int],
    thesis_style: bool,
    title_suffix: str,
) -> plt.Figure:
    """Single figure with coupling (left) and device (right) subplots."""
    fig, axes = plt.subplots(1, 2,
                             figsize=(16, 5) if not thesis_style else (18, 6))
    device = run_meta.get("cuda_device_name", run_meta.get("device", "?"))
    suptitle = f"Roofline Analysis — {device}"
    if title_suffix:
        suptitle += f"  ({title_suffix})"
    fig.suptitle(suptitle, fontsize=11 if not thesis_style else 13)

    axes[0].set_title("Coupling roofline",
                       fontsize=9 if not thesis_style else 11)
    _plot_one_roofline(axes[0], model_points, "coupling", batch_sizes_shown, thesis_style)

    axes[1].set_title("Device roofline  (per batch size)",
                       fontsize=9 if not thesis_style else 11)
    _plot_one_roofline(axes[1], model_points, "device", batch_sizes_shown, thesis_style)

    note = (
        "Coupling OI = FLOPs / (in+out) bytes/entry.  "
        "Device OI = FLOPs / (in+out + weights/B) bytes/entry.  "
        "OI uses weight payload as proxy for HBM traffic (arithmetic intensity).  "
        "Hardware: H100 NVL 94 GB (CLAIX-23 c23g), HBM3 3.938 TB/s."
    )
    fig.text(0.5, -0.03, note, ha="center", fontsize=6, style="italic",
             color="dimgrey", wrap=True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate roofline plots from model inspection JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("json_files", nargs="+", type=Path,
                   help="One or more inspection JSON result files")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: same directory as first JSON file)")
    p.add_argument("--format", dest="formats", default="pdf,png",
                   help="Output formats, comma-separated: pdf,png,svg (default: pdf,png)")
    p.add_argument("--no-coupling", action="store_true",
                   help="Skip coupling roofline figure")
    p.add_argument("--no-device", action="store_true",
                   help="Skip device roofline figure")
    p.add_argument("--combined", action="store_true",
                   help="Also write a combined single-figure with both subplots")
    p.add_argument("--batch-sizes", default=None,
                   help="Comma-separated batch sizes to plot (default: all found in data)")
    p.add_argument("--title-suffix", default="",
                   help="Extra string appended to figure titles")
    p.add_argument("--thesis", action="store_true",
                   help="Thesis-friendly style: larger fonts, tighter layout")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    run_meta, results = load_results(args.json_files)
    if not results:
        print("No model results found in input files.", file=sys.stderr)
        sys.exit(1)

    # Determine batch sizes to show
    bs_filter = None
    if args.batch_sizes:
        bs_filter = [int(x) for x in args.batch_sizes.split(",")]

    model_points = extract_points(results, batch_sizes_filter=bs_filter)

    # All batch sizes present across all models
    all_bs: list[int] = sorted({
        pt["batch_size"]
        for m in model_points
        for pt in m["points"]
    })
    batch_sizes_shown = [bs for bs in all_bs if (bs_filter is None or bs in bs_filter)]

    # Output directory
    out_dir = args.output_dir or args.json_files[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stem for output filenames
    stem = args.json_files[0].stem  # e.g. "inspection_20260810_130000_cuda"
    formats = [f.strip().lstrip(".") for f in args.formats.split(",")]

    def save_fig(fig: plt.Figure, suffix: str) -> None:
        for fmt in formats:
            out_path = out_dir / f"{stem}_{suffix}.{fmt}"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"  Written: {out_path}")
        plt.close(fig)

    print(f"Models loaded: {[m['model_id'] for m in model_points]}")
    print(f"Batch sizes:   {batch_sizes_shown}")
    print(f"Output dir:    {out_dir}")
    print()

    # Coupling roofline
    if not args.no_coupling:
        print("Plotting coupling roofline ...")
        fig = plot_coupling_roofline(model_points, run_meta, batch_sizes_shown,
                                     args.thesis, args.title_suffix)
        save_fig(fig, "roofline_coupling")

    # Device roofline
    if not args.no_device:
        print("Plotting device roofline ...")
        fig = plot_device_roofline(model_points, run_meta, batch_sizes_shown,
                                   args.thesis, args.title_suffix)
        save_fig(fig, "roofline_device")

    # Combined figure
    if args.combined:
        print("Plotting combined roofline ...")
        fig = plot_combined(model_points, run_meta, batch_sizes_shown,
                            args.thesis, args.title_suffix)
        save_fig(fig, "roofline_combined")

    print("Done.")


if __name__ == "__main__":
    main()
