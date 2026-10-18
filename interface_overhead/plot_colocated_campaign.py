#!/usr/bin/env python3
"""Create local absolute-time and logical-payload drafts from paired jobs."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
MINI_APP = HERE.parent / "mini_app"
sys.path.insert(0, str(MINI_APP))
from summarize_colocated_campaign import summarize

CONDITIONS = ("direct_nopreload", "direct_preload", "cmi_baseline",
              "cmi_profiling", "cmi_tracing")
COLORS = {"direct_nopreload": "#487797", "direct_preload": "#84AFB9",
          "cmi_baseline": "#C78645", "cmi_profiling": "#9987AD",
          "cmi_tracing": "#596B59"}
LABELS = {"direct_nopreload": "Direct", "direct_preload": "Direct\nwith preload",
          "cmi_baseline": "CMI", "cmi_profiling": "CMI\nprofiling",
          "cmi_tracing": "CMI\ntracing"}
TRAFFIC = re.compile(r"^ML_TRAFFIC input_mb=([\d.eE+-]+) output_mb=([\d.eE+-]+) "
                     r"preload_mb=([\d.eE+-]+) units=MiB scope=sum_over_ranks$", re.M)
STEPS = re.compile(r"^STEP_TIMING step=(\d+) solver=(ML|Regular) step_ms=([\d.eE+-]+)(.*)$", re.M)


def mean_ci(values):
    """Two-sided Student-t CI, df=4, across the five independent jobs."""
    if len(values) != 5 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected five finite job-level values")
    mean = statistics.mean(values)
    half = 2.7764451051977987 * statistics.stdev(values) / math.sqrt(5)
    return mean, half


def collect(root):
    result = {"campaign_root": str(root), "phase": "production",
              "confidence_interval": "Student-t, df=4; five independent job means",
              "timing_scope": "Synchronized ML-step time; first three ML calls excluded",
              "payload_scope": "Logical MiB summed over ranks, all 33 calls plus one-time preload",
              "platforms": {}, "absolute_time": [], "job_means": [], "payload": []}
    for platform in ("cpu", "gpu"):
        summary = summarize(root, platform, "production")
        if summary["excluded_jobs"] or len(summary["jobs"]) != 5:
            raise ValueError(f"{platform}: expected five complete jobs: {summary['excluded_jobs']}")
        result["platforms"][platform] = summary
        for condition in CONDITIONS:
            values = [job["mean_ml_step_ms"][condition] / 1000 for job in summary["jobs"]]
            mean, half = mean_ci(values)
            result["absolute_time"].append({"platform": platform, "condition": condition,
                "jobs": 5, "calls_per_job": 30, "mean_s": mean,
                "ci95_low_s": mean - half, "ci95_high_s": mean + half})
        for job in summary["jobs"]:
            manifest = root / job["job"] / "runs.tsv"
            with manifest.open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            for row in rows:
                condition = row["condition"]
                text = Path(row["log"]).read_text(errors="replace")
                steps = STEPS.findall(text)
                if ([int(s[0]) for s in steps] != list(range(1, 67))
                    or any(s[1] != ("ML" if int(s[0]) % 2 == 0 else "Regular") for s in steps)
                    or int(row["warmup"]) != 3 or int(row["steps"]) != 66
                    or job["measured_calls"][condition] != 30
                    or any("rank_min_ms=" not in s[3] or "rank_max_ms=" not in s[3]
                           for s in steps if s[1] == "ML")):
                    raise ValueError(f"Unexpected timing records/settings: {row['log']}")
                result["job_means"].append({"platform": platform, "job": job["job"],
                    "condition": condition, "mean_s": job["mean_ml_step_ms"][condition] / 1000,
                    "measured_calls": 30, "source_log": row["log"]})
                if condition.startswith("direct_"):
                    matches = TRAFFIC.findall(text)
                    if len(matches) != 1:
                        raise ValueError(f"Expected one logical traffic record: {row['log']}")
                    values = [float(v) for v in matches[0]]
                    if any(not math.isfinite(v) or v < 0 for v in values):
                        raise ValueError(f"Invalid traffic: {row['log']}")
                    result["payload"].append({"platform": platform, "job": job["job"],
                        "condition": condition, "input_mib": values[0], "output_mib": values[1],
                        "preload_mib": values[2], "ml_calls": 33, "source_log": row["log"]})
        for condition in CONDITIONS[:2]:
            payload = [r for r in result["payload"] if r["platform"] == platform
                       and r["condition"] == condition]
            for field in ("input_mib", "output_mib", "preload_mib"):
                if len({r[field] for r in payload}) != 1:
                    raise ValueError(f"Nonidentical logical payloads: {platform}/{condition}/{field}")
    return result


def export(data, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "measurement_summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    paired = [{"platform": platform, **row} for platform, summary in data["platforms"].items()
              for row in summary["paired_comparisons"]]
    for name, rows in (("absolute_time", data["absolute_time"]), ("job_means", data["job_means"]),
                       ("payload", data["payload"]), ("paired_comparisons", paired)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def render(data, output, style_path):
    sys.path.insert(0, str(style_path))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    interface_conditions = (CONDITIONS[0], *CONDITIONS[2:])
    for name, conditions, platforms in (
        ("interface_scorep_cpu", interface_conditions, ("cpu",)),
        ("interface_scorep_gpu", interface_conditions, ("gpu",)),
        ("preload_time", CONDITIONS[:2], ("cpu", "gpu")),
    ):
        compact = len(platforms) == 1
        fig, axes = ts.figure(width=0.49 if compact else 1.0,
                              ncols=len(platforms), height_in=1.8 if compact else 3.55)
        if compact:
            axes = [axes]
            fig.subplots_adjust(left=0.23, right=0.99, bottom=0.23, top=0.98)
        else:
            fig.subplots_adjust(left=0.12, right=0.99, bottom=0.25, top=0.87, wspace=0.45)
        for ax, platform in zip(axes, platforms):
            rows = [next(r for r in data["absolute_time"] if r["platform"] == platform
                         and r["condition"] == condition) for condition in conditions]
            means = [r["mean_s"] for r in rows]
            errors = [r["ci95_high_s"] - r["mean_s"] for r in rows]
            ax.bar(range(len(rows)), means, width=0.64, yerr=errors, capsize=3,
                   color=[COLORS[c] for c in conditions], edgecolor="white", linewidth=0.6,
                   error_kw={"elinewidth": 0.9, "capthick": 0.9}, zorder=3)
            upper = max(m + e for m, e in zip(means, errors))
            for i, (mean, error) in enumerate(zip(means, errors)):
                ax.text(i, mean + error + upper * 0.025, f"{mean:.2f}", ha="center",
                        va="bottom", fontsize=8)
            ax.set_xticks(range(len(rows)), [LABELS[c] for c in conditions], fontsize=9)
            ax.set_ylim(0, upper * (1.28 if compact else 1.17))
            if not compact:
                ax.set_title(platform.upper())
            ax.set_ylabel("Mean ML-step\ntime [s]" if compact else "Mean ML-step time [s]", fontsize=10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3 if compact else 5))
            ax.grid(False, axis="x")
            ax.grid(axis="y", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        if not compact:
            fig.text(0.5, 0.045, "Steady state; 95% CI across five job means", ha="center", fontsize=9)
        ts.save(fig, output / name)

    fig, axes = ts.figure(ncols=2, height_in=3.65)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.27, top=0.79, wspace=0.47)
    for ax, platform in zip(axes, ("cpu", "gpu")):
        maximum = 0
        for index, condition in enumerate(CONDITIONS[:2]):
            row = next(r for r in data["payload"] if r["platform"] == platform
                       and r["condition"] == condition)
            values = [row[f"{kind}_mib"] for kind in ("input", "output", "preload")]
            positions = [x + (index - 0.5) * 0.36 for x in range(3)]
            ax.bar(positions, values, width=0.34, color=COLORS[condition],
                   label=("Without preload" if index == 0 else "With preload"), zorder=3)
            maximum = max(maximum, *values)
            # Vertical labels keep small/zero payloads readable on a linear axis.
            for x, value in zip(positions, values):
                ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            rotation=90, fontsize=8)
        ax.set_ylim(0, maximum * 1.38)
        ax.set_xticks(range(3), ["Input", "Output", "Terrain\npreload"], fontsize=9)
        ax.set_ylabel("Logical payload [MiB]", fontsize=10)
        ax.set_title(platform.upper())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.55, 0.98), fontsize=9)
    fig.text(0.5, 0.065, "Full run: 33 ML calls; summed over 96 ranks", ha="center", fontsize=9)
    ts.save(fig, output / "preload_payload")
    for name in ("interface_scorep_cpu", "interface_scorep_gpu", "preload_time", "preload_payload"):
        for suffix in ("png", "pdf", "pgf"):
            path = output / f"{name}.{suffix}"
            if not path.exists() or not path.stat().st_size:
                raise RuntimeError(f"Missing figure export: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path,
        default=MINI_APP / "logs/colocated_campaign/colocated_full_4961357")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    parser.add_argument("--style-path", type=Path,
        default=Path.home() / "Master-Thesis/gitlab/sourcecode")
    args = parser.parse_args()
    measurements = collect(args.campaign_root.resolve())
    export(measurements, args.output)
    render(measurements, args.output, args.style_path)
    print(f"Generated CPU/GPU subfigures, preload drafts and summaries in {args.output.resolve()}")
