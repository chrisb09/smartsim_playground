"""Figures: measured bar charts, HBM/L2 traffic, roofline, and throughput plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox, offset_copy

MODEL_COLORS = {"watercnn": "#0072B2", "mmcp_test_mlp_m5": "#009E73", "tbl_transformer": "#882255", "giant_mlp": "#D55E00"}
MEASURED_COLOR = "#EE6677"
L2_COLOR = "#CCBB44"
PRECISION_COLORS = {"fp32": "#4477AA", "tf32": "#EE6677", "fp16": "#228833", "bf16": "#CCBB44", "fp8": "#66CCEE", "int8": "#AA3377", "fp64": "#BBBBBB"}
PRECISION_ORDER = ("fp32", "tf32", "fp16", "bf16", "fp8", "int8", "fp64")
FALLBACK_COMPUTE_GFLOPS = {"fp32": 60320.0, "tf32": 482600.0}
SINGLE_FIGSIZE = (6.2, 4.6)
PAIR_FIGSIZE = (3.4, 2.9)


def read_table(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def collect(table: list[dict], metric: str) -> dict[tuple[str, str, int], float]:
    values: dict[tuple[str, str, int], float] = {}
    for row in table:
        if row["metric"] != metric or row["value"] in ("", None):
            continue
        values[(row["model_id"], row["precision"], int(row["batch_size"]))] = float(row["value"])
    return values


def _display_names(table: list[dict]) -> dict[str, str]:
    return {row["model_id"]: row["display_name"] for row in table}


def _human(value: float, unit: str) -> str:
    if unit == "bytes":
        for scale, suffix in ((1e9, " GB"), (1e6, " MB"), (1e3, " kB")):
            if abs(value) >= scale:
                return f"{value / scale:.2f}{suffix}"
        return f"{value:.0f} B"
    if unit == "flops":
        for scale, suffix in ((1e12, " TFLOP"), (1e9, " GFLOP"), (1e6, " MFLOP"), (1e3, " kFLOP")):
            if abs(value) >= scale:
                return f"{value / scale:.2f}{suffix}"
        return f"{value:.0f} FLOP"
    return f"{value:,.4g}"


def _finish(fig, run_dir: Path, name: str) -> Path:
    output_dir = run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pdf"
    fig.tight_layout(); fig.savefig(path); fig.savefig(output_dir / f"{name}.png"); plt.close(fig)
    return path


def bar_measured(run_dir: Path, table: list[dict], metric: str, unit: str, title: str, ylabel: str,
                 name: str, comparison_batch: int) -> Path | None:
    values = collect(table, metric)
    models = sorted({key[0] for key in values})
    if not models:
        return None
    displays = _display_names(table)
    fig, ax = plt.subplots(figsize=PAIR_FIGSIZE, dpi=250)
    for index, model in enumerate(models):
        value = values.get((model, "fp32", comparison_batch))
        if not value:
            continue
        ax.bar(index, value, 0.5, color=MEASURED_COLOR)
        ax.annotate(_human(value, unit), (index, value), ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([displays.get(model, model) for model in models])
    positives = [value for value in values.values() if value and value > 0]
    if positives:
        ax.set_ylim(min(positives) * 0.3, max(positives) * 8)
    ax.set(ylabel=ylabel)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"{title} (B={comparison_batch})", fontsize=8.5)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    return _finish(fig, run_dir, name)


def l2_cache_bars(run_dir: Path, cache_table: list[dict], comparison_batch: int, name: str = "bar_l2_vs_dram") -> Path | None:
    dram = collect(cache_table, "dram_total")
    l2 = collect(cache_table, "l2_total")
    models = sorted({key[0] for key in dram} | {key[0] for key in l2})
    if not models:
        return None
    displays = _display_names(cache_table)
    xs = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=PAIR_FIGSIZE, dpi=250)
    for index, model in enumerate(models):
        dram_value = dram.get((model, "fp32", comparison_batch))
        l2_value = l2.get((model, "fp32", comparison_batch))
        if dram_value:
            ax.bar(index - width / 2, dram_value, width, color=MEASURED_COLOR, label="RAM" if index == 0 else None)
            ax.annotate(_human(dram_value, "bytes"), (index - width / 2, dram_value), ha="center", va="bottom", fontsize=7)
        if l2_value:
            ax.bar(index + width / 2, l2_value, width, color=L2_COLOR, label="L2 requested" if index == 0 else None)
            ax.annotate(_human(l2_value, "bytes"), (index + width / 2, l2_value), ha="center", va="bottom", fontsize=7)
        if dram_value and l2_value:
            ax.annotate(f"L2/RAM {l2_value / dram_value:,.0f}x", (index, max(dram_value, l2_value)),
                        xytext=(0, 14), textcoords="offset points", ha="center", fontsize=6.5, color="#333333")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([displays.get(model, model) for model in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    positives = [value for values in (dram, l2) for value in values.values() if value and value > 0]
    if positives:
        ax.set_ylim(min(positives) * 0.3, max(positives) * 12)
    ax.set(ylabel="Bytes (log scale)")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"L2 versus RAM (B={comparison_batch})", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    return _finish(fig, run_dir, name)


def bar_precision(run_dir: Path, precision_table: list[dict], comparison_batch: int, name: str = "bar_precision") -> Path | None:
    """Stacked 100% bars: share of counter-derived FLOPs per execution precision."""
    per_model: dict[str, dict[str, float]] = {}
    for row in precision_table:
        if row["precision"] != "fp32" or int(row["batch_size"]) != comparison_batch or row["value"] in ("", None):
            continue
        category = row["metric"].removesuffix("_flops")
        if category not in PRECISION_ORDER:
            continue
        per_model.setdefault(row["model_id"], {})[category] = float(row["value"])
    models = sorted(per_model)
    if not models:
        return None
    displays = _display_names(precision_table)
    categories = [category for category in PRECISION_ORDER
                  if any(per_model[model].get(category, 0.0) > 0 for model in models)]
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=250)
    labelled: set[str] = set()
    for index, model in enumerate(models):
        totals = sum(per_model[model].values()) or 1.0
        bottom = 0.0
        for category in categories:
            share = 100.0 * per_model[model].get(category, 0.0) / totals
            if share <= 0:
                continue
            ax.bar(index, share, 0.55, bottom=bottom, color=PRECISION_COLORS.get(category, "#333333"),
                   label=category.upper() if category not in labelled else None)
            labelled.add(category)
            if share >= 4:
                ax.annotate(f"{share:.0f}%", (index, bottom + share / 2), ha="center", va="center", fontsize=7.5, color="white")
            bottom += share
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([displays.get(model, model) for model in models])
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, 100)
    ax.set(ylabel="Share of counter-derived FLOPs (%)", title=f"Arithmetic precision composition (B={comparison_batch})")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    return _finish(fig, run_dir, name)


def _batch_label(batch: int) -> str:
    exponent = batch.bit_length() - 1
    if batch > 0 and 2 ** exponent == batch:
        return f"$B=2^{{{exponent}}}$"
    return f"B={batch}"


def _place_point_labels(ax, entries: list[tuple[float, float, str, str]],
                        fontsize: float = 6.5, pad: float = 1.5) -> int:
    """Place labels next to points, avoiding overlaps with each other, markers, and the legend.

    ``entries`` are ``(x, y, text, color)`` in data coordinates. Candidate offsets
    are tried from small to large; labels pushed farther than 9 pt get a thin
    leader line. Returns the number of labels that could not be placed without
    overlap.
    """
    if not entries:
        return 0
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    occupied: list[Bbox] = []
    legend = ax.get_legend()
    if legend is not None:
        occupied.append(legend.get_window_extent(renderer))
    positions = ax.transData.transform([(x, y) for x, y, _, _ in entries])
    marker_radius = 4.0 * fig.dpi / 72.0
    markers = [Bbox.from_extents(px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius)
               for px, py in positions]
    density = [int(np.sum(np.hypot(*(positions - position).T) < 45.0) - 1) for position in positions]
    order = sorted(range(len(entries)), key=lambda index: (density[index], index))
    angles = [index * math.pi / 8 for index in range(16)]
    radii = (4.0, 7.0, 10.5, 15.0, 21.0, 28.0, 38.0, 50.0)
    fallback = 0

    def probe(x: float, y: float, text: str, dx: float, dy: float) -> Bbox:
        transform = offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
        artist = Text(x, y, text, transform=transform, fontsize=fontsize, ha="center", va="center", figure=fig)
        bbox = artist.get_window_extent(renderer)
        return Bbox.from_extents(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad)

    for index in order:
        x, y, text, color = entries[index]
        chosen: tuple[float, float, Bbox] | None = None
        for radius in radii:
            for angle in angles:
                dx, dy = radius * math.cos(angle), radius * math.sin(angle)
                bbox = probe(x, y, text, dx, dy)
                inside = (bbox.x0 >= axes_bbox.x0 and bbox.x1 <= axes_bbox.x1
                          and bbox.y0 >= axes_bbox.y0 and bbox.y1 <= axes_bbox.y1)
                if inside and not any(bbox.overlaps(other) for other in occupied + markers):
                    chosen = (dx, dy, bbox)
                    break
            if chosen is not None:
                break
        if chosen is None:
            dx, dy, bbox = 4.0, 4.0, probe(x, y, text, 4.0, 4.0)
            fallback += 1
        else:
            dx, dy, bbox = chosen
        arrow = ({"arrowstyle": "-", "color": color, "lw": 0.4, "shrinkA": 0, "shrinkB": 1.5,
                  "alpha": 0.85, "zorder": 1.5} if math.hypot(dx, dy) > 9.0 else None)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                    color=color, ha="center", va="center", zorder=6, arrowprops=arrow)
        occupied.append(bbox)
    return fallback


def roofline_measured(run_dir: Path, oi_table: list[dict], flops_table: list[dict], timing_table: list[dict],
                      hardware: dict, include_tf32: bool = False, name: str = "roofline_hbm3_measured") -> Path | None:
    oi = collect(oi_table, "operational_intensity")
    flops = collect(flops_table, "measured_flops")
    resident = collect(timing_table, "resident_forward_ms")
    if not oi:
        return None
    hbm = hardware.get("hbm_gbps", 3938.0)
    peaks = hardware.get("compute_gflops", FALLBACK_COMPUTE_GFLOPS)
    present = {row["precision"] for row in oi_table}
    displays = _display_names(oi_table)
    series: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for model in sorted({key[0] for key in oi}):
        for precision in sorted({key[1] for key in oi if key[0] == model}):
            points = []
            for (candidate, candidate_precision, batch), intensity in sorted(oi.items()):
                if candidate != model or candidate_precision != precision:
                    continue
                key = (model, precision, batch)
                measured = flops.get(key)
                milliseconds = resident.get(key)
                if not measured or not milliseconds:
                    continue
                points.append((batch, intensity, measured / (milliseconds / 1000) / 1e9))
            if points:
                points.sort()
                series[(model, precision)] = points
    if not series:
        return None
    xs_values = [point[1] for points in series.values() for point in points]
    ys_values = [point[2] for points in series.values() for point in points]
    x_lo, x_hi = min(xs_values) * 0.3, max(xs_values) * 3
    xs = np.logspace(np.log10(x_lo), np.log10(x_hi), 600)
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=250)
    ceilings = []
    if "fp32" in peaks and "fp32" in present:
        ceilings.append(("fp32", peaks["fp32"], "#444444", "--"))
    if include_tf32 and "tf32" in peaks:
        ceilings.append(("tf32", peaks["tf32"], "#999999", "-."))
    for precision, peak, color, style in ceilings:
        ax.plot(xs, np.minimum(peak, hbm * xs), color=color, linestyle=style,
                label=f"{precision.upper()} ceiling ({peak / 1000:.1f} TFLOP/s, {hbm / 1000:.3f} TB/s)")
    labels: list[tuple[float, float, str, str]] = []
    for (model, precision), points in series.items():
        color = MODEL_COLORS.get(model, "#333333")
        ax.plot([p[1] for p in points], [p[2] for p in points], "o-", color=color,
                label=f"{displays.get(model, model)} ({precision.upper()})")
        labels.extend((intensity, performance, _batch_label(batch), color)
                      for batch, intensity, performance in points)
    ax.set(xscale="log", yscale="log",
           xlabel="Measured operational intensity (FLOP / NCU RAM byte)",
           ylabel="GFLOP/s (FMA + tensor cores)",
           title="Measured HBM3 Roofline (NCU counters, CUDA-event timing)")
    ax.set_xlim(x_lo, x_hi)
    y_top = max(ys_values) * 2
    if any(precision == "tf32" for precision, *_ in ceilings):
        y_top = max(y_top, max(peaks["tf32"], 1.0) * 1.2)
    if min(ys_values) > 0:
        ax.set_ylim(min(ys_values) * 0.5, y_top)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fallback = _place_point_labels(ax, labels)
    if fallback:
        print(f"  roofline labels: {fallback}/{len(labels)} could not be placed without overlap")
    return _finish(fig, run_dir, name)


def throughput(run_dir: Path, timing_table: list[dict], precision: str, hardware: dict) -> Path | None:
    serial = collect(timing_table, "serial_samples_per_s")
    rows = [key for key in serial if key[1] == precision]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=250)
    displays = _display_names(timing_table)
    for model in sorted({key[0] for key in rows}):
        pairs = sorted((key[2], serial[key]) for key in rows if key[0] == model)
        color = MODEL_COLORS.get(model, "#333333")
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], "o-", color=color, label=displays.get(model, model))
    ax.set(xscale="log", yscale="log", xlabel="Batch size", ylabel="Serial end-to-end throughput (samples/s)",
           title=f"Serial throughput vs batch size ({precision.upper()}, H2D + forward + D2H)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, run_dir, f"throughput_{precision}")


def hbm_vs_batch(run_dir: Path, hbm_table: list[dict], name: str = "hbm_vs_batch") -> Path | None:
    read = collect(hbm_table, "read")
    write = collect(hbm_table, "write")
    models = sorted({key[0] for key in read} | {key[0] for key in write})
    if not models:
        return None
    displays = _display_names(hbm_table)
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=250)
    for model in models:
        color = MODEL_COLORS.get(model, "#333333")
        read_points = sorted((key[2], value) for key, value in read.items() if key[0] == model and value > 0)
        write_points = sorted((key[2], value) for key, value in write.items() if key[0] == model and value > 0)
        if read_points:
            ax.plot([p[0] for p in read_points], [p[1] for p in read_points], "o-", color=color, label=f"{displays.get(model, model)} read")
        if write_points:
            ax.plot([p[0] for p in write_points], [p[1] for p in write_points], "s--", color=color, alpha=0.7, label=f"{displays.get(model, model)} write")
    ax.set(xscale="log", yscale="log", xlabel="Batch size", ylabel="NCU RAM bytes",
           title="Measured HBM3 RAM traffic versus batch size (FP32)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7)
    return _finish(fig, run_dir, name)


def oi_vs_batch(run_dir: Path, oi_table: list[dict], name: str = "oi_vs_batch") -> Path | None:
    oi = collect(oi_table, "operational_intensity")
    if not oi:
        return None
    displays = _display_names(oi_table)
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=250)
    for model in sorted({key[0] for key in oi}):
        points = sorted((key[2], value) for key, value in oi.items() if key[0] == model)
        color = MODEL_COLORS.get(model, "#333333")
        ax.plot([p[0] for p in points], [p[1] for p in points], "o-", color=color, label=displays.get(model, model))
    ax.set(xscale="log", yscale="log", xlabel="Batch size", ylabel="Measured operational intensity (FLOP / RAM byte)",
           title="Measured operational intensity versus batch size (FP32)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, run_dir, name)
