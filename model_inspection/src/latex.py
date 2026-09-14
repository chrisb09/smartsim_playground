"""Render the LaTeX report from measured CSV tables and figures.

The template lives in ``src/templates/report.tex.tpl``. Rendering never
fabricates values: missing measurements appear as ``n/a``. Compilation with
``latexmk`` is optional and controlled by the CLI.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

from catalog import spec_for

TEMPLATE = Path(__file__).resolve().parent / "templates" / "report.tex.tpl"
FALLBACK_LATEXMK = Path.home() / "opt/texlive/2026/bin/x86_64-linux/latexmk"


def load_tables(tables_dir: Path) -> dict[str, list[dict]]:
    tables = {}
    for path in sorted(tables_dir.glob("*.csv")):
        with path.open(newline="") as handle:
            tables[path.stem] = list(csv.DictReader(handle))
    return tables


def lookup(rows: list[dict], model_id: str, precision: str, batch: int, metric: str) -> float | None:
    for row in rows:
        if row["model_id"] == model_id and row["precision"] == precision and int(row["batch_size"]) == batch and row["metric"] == metric:
            if row["value"] in ("", None):
                return None
            return float(row["value"])
    return None


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex_escape(text: str) -> str:
    """Escape dynamic text for LaTeX text mode (single pass)."""
    return re.sub(r"[\\&%$#_{}~^]", lambda match: _LATEX_ESCAPES[match.group(0)], str(text))


def num(value: float | None, digits: int = 0) -> str:
    """Text-mode number for prose and captions."""
    if value is None:
        return "n/a"
    if digits:
        return f"\\num{{{value:.{digits}f}}}"
    return f"\\num{{{value:.0f}}}"


def cell(value: float | None, digits: int | None = None) -> str:
    """Bare number for siunitx S columns (alignment and grouping by siunitx)."""
    if value is None:
        return "{n/a}"
    if digits is not None:
        return f"{value:.{digits}f}"
    return f"{value:.0f}"


def shape_text(fields: list[dict]) -> str:
    shapes = ["[" + ",".join("B" if dim is None else str(dim) for dim in field["shape"]) + "]" for field in fields]
    if len(shapes) == 1:
        return "\\texttt{" + shapes[0] + "}"
    if len(set(shapes)) == 1:
        return f"{len(shapes)}$\\times$\\texttt{{{shapes[0]}}}"
    return ", ".join("\\texttt{" + shape + "}" for shape in shapes)


def model_table(catalog: dict, models: list[str], storage_t: list[dict]) -> str:
    body = [
        "\\small",
        "\\begin{tabular}{@{}p{4.6cm} p{2.9cm} p{2.4cm} S[table-format=10.0]@{}}",
        "\\toprule",
        "Model & Inputs & Outputs & {Parameters} \\\\",
        "\\midrule",
    ]
    footnotes = []
    note_index = 0
    for model_id in models:
        spec = spec_for(catalog, model_id)
        long_name = tex_escape(spec.get("long_name") or spec.get("display_name", model_id))
        notes = tex_escape(spec.get("notes", ""))
        mark = ""
        if notes:
            note_index += 1
            mark = f"\\footnotemark[{note_index}]"
            footnotes.append(f"\\footnotetext[{note_index}]{{{notes}}}")
        inputs = shape_text(spec["inputs"])
        outputs = shape_text(spec["outputs"])
        parameters = lookup(storage_t, model_id, "all", 0, "parameter_elements")
        body.append(f"\\textbf{{{long_name}}}{mark} & {inputs} & {outputs} & {cell(parameters)} \\\\")
    body.extend(["\\bottomrule", "\\end{tabular}"])
    tabular = "\n".join(body)
    table = ("\\begin{table}[h]\n\\centering\n" + tabular + "\n"
             "\\caption{Measured models: tensor contracts and parameter counts.}\n\\label{tab:models}\n\\end{table}")
    if footnotes:
        table += "\n\n" + "\n".join(footnotes)
    return table


def group_text(catalog: dict, models: list[str]) -> str:
    groups = catalog.get("groups", {})
    used = []
    for model_id in models:
        group = spec_for(catalog, model_id).get("group")
        if group and group not in used:
            used.append(group)
    paragraphs = []
    for group in used:
        entry = groups.get(group, {})
        long_name = tex_escape(entry.get("long_name", group))
        explanation = tex_escape(entry.get("explanation", ""))
        paragraphs.append(f"\\paragraph{{{tex_escape(group)} ({long_name}):}} {explanation}")
    return "\n\n".join(paragraphs)


def measured_table(catalog: dict, models: list[str], flops_t, hbm_t, pcie_t, precision: str, batch: int,
                   caption: str, label: str, resize: bool = False) -> str:
    body = [
        "\\begin{tabular}{l S[table-format=15.0] S[table-format=15.0] S[table-format=10.0] S[table-format=4.3] S[table-format=4.3]}",
        "\\toprule",
        "Model & {FLOPs} & {RAM bytes} & {PCIe bytes} & {FLOP/RAM byte} & {FLOP/PCIe byte} \\\\",
        "\\midrule",
    ]
    for model_id in models:
        spec = spec_for(catalog, model_id)
        display = tex_escape(spec.get("short_name") or spec.get("display_name", model_id))
        f = lookup(flops_t, model_id, precision, batch, "measured_flops")
        h = lookup(hbm_t, model_id, precision, batch, "total")
        p = lookup(pcie_t, model_id, precision, batch, "total")
        o_ram = f / h if f and h else None
        o_pcie = f / p if f and p else None
        body.append(f"{display} & {cell(f)} & {cell(h)} & {cell(p)} & {cell(o_ram, 3)} & {cell(o_pcie, 3)} \\\\")
    body.extend(["\\bottomrule", "\\end{tabular}"])
    tabular = "\n".join(body)
    if resize:
        tabular = "\\resizebox{\\linewidth}{!}{%\n" + tabular + "\n}"
    return ("\\begin{table}[h]\n\\centering\n" + tabular + "\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}")


def render(run_dir: Path, catalog: dict, models: list[str], precisions: list[str], hardware: dict,
           comparison_batch: int, provenance: dict) -> Path:
    tables = load_tables(run_dir / "tables")
    required = ("model_storage", "flops_measured_hardware", "hbm3_measured", "pcie_measured",
                "oi_hbm3_measured", "timing_measured")
    for name in required:
        if name not in tables:
            raise FileNotFoundError(f"missing table {name}; run the table stage first")
    precision = "fp32" if "fp32" in precisions else precisions[0]
    gpu = provenance.get("gpu", {})
    gpu_name = gpu.get("name", "unknown GPU")
    hostname = provenance.get("hostname", "unknown host")
    ncu_version = provenance.get("ncu_version", "unknown")
    cache_control = provenance.get("ncu_cache_control", "unknown")
    replay_mode = provenance.get("ncu_replay_mode", "unknown")
    ladders = provenance.get("ladders", {})
    nsys_note = ("Per-configuration Nsight Systems traces provide the measured PCIe payload."
                 if provenance.get("nsys_collected") else "Nsight Systems PCIe collection was not part of this run.")
    ladder_items = "\n".join(f"\\item \\textbf{{{tex_escape(spec_for(catalog, model).get('short_name', model))}}}: {batches}"
                             for model, batches in sorted(ladders.items()))
    ladder_list = "\\begin{itemize}\n" + ladder_items + "\n\\end{itemize}"
    device_extra = ""
    if gpu.get("multi_processor_count"):
        device_extra = f" The device reports {gpu['multi_processor_count']} SMs"
        if gpu.get("max_sm_clock_mhz"):
            device_extra += f" and a {gpu['max_sm_clock_mhz']} MHz maximum SM clock"
        device_extra += "."

    overview = (
        f"This measurement-only run ({tex_escape(provenance.get('run_name', run_dir.name))}) profiled {len(models)} models "
        f"on {tex_escape(gpu_name)} at {tex_escape(hostname)}.{device_extra} "
        f"Precision: {', '.join(precisions)}. Nsight Compute version: {tex_escape(ncu_version)}. {nsys_note} "
        "The discovered batch ladders are:"
    )
    methodology_arithmetic = (
        "All reported quantities are measured directly from hardware counters and activity traces. "
        "Arithmetic comes from Nsight Compute hardware counters: executed FFMA, FADD, and FMUL thread instructions plus "
        "tensor-core operations (TF32 precision, FP32 accumulation). All FLOP figures in this report are counter-derived and computed as "
        "$\\text{FLOPs} = 2\\cdot\\text{FFMA} + \\text{FADD} + \\text{FMUL} + \\text{tensor ops}$; the tensor counters "
        "already count one multiply and one add per MAC (validated against a GEMM of known operation count), so they enter "
        "unweighted. These are executed hardware "
        "work rather than a mathematical operation count; algorithm choice and fused kernels change the number without "
        "changing the computation. Tensor-path counters include tiling and padding overhead, so the tensor contribution is "
        "a coarse upper bound rather than an exact operation count."
    )
    methodology_memory = (
        "Memory traffic comes from Nsight Compute device-memory counters ($\\texttt{dram\\_\\_bytes}$ read/write, physically "
        "HBM3) and L2 request counters ($\\texttt{lts\\_\\_t\\_\\_bytes}$, L2 read/write sectors). Counters are collected with "
        f"warm caches ($\\texttt{{cache-control}} = \\texttt{{{tex_escape(cache_control)}}}$, "
        f"$\\texttt{{replay-mode}} = \\texttt{{{tex_escape(replay_mode)}}}$): "
        "the model has already run before profiling, so its code, constants, and (if they fit) weights are L2-resident. "
        "RAM writes are a lower bound because write-back caches can retain outputs during the measurement window."
    )
    methodology_transfer = (
        "PCIe payload comes from Nsight Systems CUPTI memcpy activity records restricted to the NVTX-marked audit window, "
        "so one-time weight uploads and warmup copies are excluded and only per-inference H2D/D2H payload is counted."
    )
    methodology_ladders = (
        "Batch sizes are discovered per model by an adaptive scheme: starting from $B=1$, the next candidate is the current "
        "largest times the ladder multiplier (for example $\\times 10$), and a two-point line through the measured peak VRAM "
        "predicts the candidate. Growth stops once the prediction exceeds 90\\% of device memory. The internal fit is used "
        "only as a safety check and is not reported."
    )
    results = (
        f"Tables \\ref{{tab:measured-b1}} and \\ref{{tab:measured-b}} list the measured counters at $B=1$ and "
        f"$B={comparison_batch}$. The figures show arithmetic, RAM and L2 traffic, the measured roofline, operational "
        "intensity versus batch size, and throughput. The two intensity columns divide FLOPs by the measured RAM bytes and "
        "by the measured PCIe payload, respectively."
    )
    limitations = (
        "All values are measurements of this specific execution on this specific GPU and allocation. "
        "\\textbf{FLOPs} count executed instructions, not mathematical operations: algorithm choice, "
        "predication, and fused kernels change the number without changing the computation. The roofline y-axis mixes "
        "FMA-pipe and tensor-pipe work; the FMA share stays below the FP32 ceiling, while totals can exceed it when tensor "
        "cores contribute. No combined FP32+tensor-core ceiling is drawn because the pipes share instruction issue and memory "
        "bandwidth, so their peaks are not additive. "
        "\\textbf{RAM reads} are measured with warm caches, so small-batch traffic can be near zero once code, constants, "
        "and weights fit in the 50\\,MB L2; measured operational intensity therefore peaks at small batches and falls to the "
        "asymptotic per-sample balance as activations outgrow the cache. This is a cache-capacity effect, not a dependency "
        "of intensity on batch size. "
        "\\textbf{RAM writes} are a lower bound because write-back caches can retain outputs during the measurement window. "
        "\\textbf{L2 request bytes} count cache accesses, not RAM traffic; an L2/RAM ratio above one indicates reuse. "
        "Application replay re-runs the process once per metric pass, which also suppresses normal kernel overlap and some "
        "host-side effects. Measured PCIe bytes are CUPTI memcpy payloads, not wire-level protocol traffic. "
        "Values at small batches on shared login GPUs can be perturbed by other tenants' work; exclusive allocations are "
        "the reference environment."
    )

    figures_dir = run_dir / "figures"
    single_captions = {
        "bar_precision": "Arithmetic precision composition at the comparison batch: share of counter-derived FLOPs executed on "
                         "the FP32 FMA pipe versus tensor cores (TF32 and other precisions).",
        "roofline_hbm3_measured": "Measured roofline: FLOPs per measured RAM byte versus measured FLOP/s (FMA + tensor cores). "
                                  "The FP32 and (when present) tensor-core (TF32) ceilings are reference lines.",
        "hbm_vs_batch": "Measured RAM read/write traffic versus batch size; zero-byte points (L2-resident reads, "
                        "un-evicted writes) are omitted.",
        "oi_vs_batch": "Measured operational intensity versus batch size; the decline marks the transition from L2-resident "
                       "to RAM-bound execution.",
        "throughput_fp32": "Serial end-to-end throughput vs batch size (FP32).",
        "throughput_tf32": "Serial end-to-end throughput vs batch size (TF32).",
    }
    pair_captions = [
        (["bar_flops", "bar_hbm3"], "Measured FLOPs (left) and HBM3 RAM traffic (right) at the comparison batch."),
        (["bar_pcie", "bar_l2_vs_dram"], "Measured PCIe payload (left) and L2-versus-RAM traffic (right) at the comparison batch."),
    ]

    def figure_block(stems: list[str], caption: str, width: str) -> str:
        includes = "\n".join(
            f"\\begin{{minipage}}{{{width}\\textwidth}}\\centering"
            f"\\includegraphics[width=\\linewidth]{{{stem}.pdf}}\\end{{minipage}}"
            + ("\\hfill" if index < len(stems) - 1 else "")
            for index, stem in enumerate(stems))
        return "\\begin{figure}[h]\n\\centering\n" + includes + f"\n\\caption{{{caption}}}\n\\end{{figure}}"

    figure_blocks = []
    for stems, caption in pair_captions:
        present = [stem for stem in stems if (figures_dir / f"{stem}.pdf").exists()]
        if not present:
            continue
        if len(present) == 2:
            figure_blocks.append(figure_block(present, caption, "0.49"))
        else:
            figure_blocks.append(figure_block(present, single_captions.get(present[0], ""), "0.85"))
    for stem, caption in single_captions.items():
        if (figures_dir / f"{stem}.pdf").exists():
            figure_blocks.append(figure_block([stem], caption, "0.85"))
    figures_text = "\n\n".join(figure_blocks) if figure_blocks else "No figures were generated for this run."

    replacements = {
        "date": provenance.get("date", ""),
        "model_count": str(len(models)),
        "overview_text": overview,
        "ladder_list": ladder_list,
        "group_text": group_text(catalog, models),
        "model_table": model_table(catalog, models, tables["model_storage"]),
        "methodology_arithmetic": methodology_arithmetic,
        "methodology_memory": methodology_memory,
        "methodology_transfer": methodology_transfer,
        "methodology_ladders": methodology_ladders,
        "results_text": results,
        "limitations_text": limitations,
        "comparison_batch": str(comparison_batch),
        "measured_b1_table": measured_table(catalog, models, tables["flops_measured_hardware"], tables["hbm3_measured"],
                                            tables["pcie_measured"], precision, 1,
                                            "Measured counters at $B=1$.", "tab:measured-b1", resize=True),
        "measured_b_table": measured_table(catalog, models, tables["flops_measured_hardware"], tables["hbm3_measured"],
                                           tables["pcie_measured"], precision, comparison_batch,
                                           f"Measured counters at $B={comparison_batch}$.", "tab:measured-b", resize=True),
        "figures_text": figures_text,
    }
    rendered = TEMPLATE.read_text()
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tex_path = report_dir / "report.tex"
    tex_path.write_text(rendered)
    for pdf in figures_dir.glob("*.pdf"):
        shutil.copy2(pdf, report_dir / pdf.name)
    return tex_path


def find_latexmk(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    found = shutil.which("latexmk")
    if found:
        return Path(found)
    return FALLBACK_LATEXMK if FALLBACK_LATEXMK.exists() else None


def compile_pdf(report_dir: Path, latexmk: str | None, logger, texlive_bin: str | None = None) -> Path | None:
    binary = find_latexmk(latexmk)
    if binary is None:
        logger.warning("latexmk not found; skipping PDF compilation")
        return None
    command = [str(binary), "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"]
    env = dict(os.environ)
    # Ensure the toolchain matching latexmk (or an explicitly configured TeX
    # installation) wins over any system TeX installation on PATH.
    tex_path = Path(texlive_bin).expanduser() if texlive_bin else binary.parent
    env["PATH"] = str(tex_path) + os.pathsep + env.get("PATH", "")
    logger.info("$ " + " ".join(command))
    result = subprocess.run(command, cwd=report_dir, capture_output=True, text=True, timeout=900, env=env)
    if result.returncode != 0:
        tail = "\n".join((result.stdout or "").splitlines()[-15:])
        logger.error(f"latexmk failed ({result.returncode}):\n{tail}")
        return None
    pdf = report_dir / "report.pdf"
    if pdf.exists():
        logger.info(f"Compiled {pdf}")
        return pdf
    return None
