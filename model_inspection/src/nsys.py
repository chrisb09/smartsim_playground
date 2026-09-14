"""Nsight Systems collection: one trace per model/precision/batch, CUPTI memcpy totals.

Each trace profiles repeated explicit H2D -> forward -> D2H steps for exactly
one batch size, so the per-config memcpy payload can be read directly from the
exported SQLite without NVTX-to-record timestamp correlation. The Activity
records are the exact bytes actually copied, independent of the CUDA-event
timings.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from catalog import schema_bytes, spec_for

COPY_KIND_HTOD = 1
COPY_KIND_DTOH = 2


def memcpy_totals(sqlite_path: Path) -> dict:
    """Sum memcpy payload by direction, restricted to the outer NVTX audit range.

    Without the window, one-time costs such as loading weights onto the GPU and
    the warmup input copy would be counted as per-inference PCIe traffic. The
    window makes the measured value comparable to the batch-size-scaled tensor
    accounting.
    """
    totals = {"h2d_bytes": 0, "d2h_bytes": 0, "other_bytes": 0, "h2d_count": 0, "d2h_count": 0, "other_count": 0, "window": "full_trace"}
    with sqlite3.connect(sqlite_path) as connection:
        window = connection.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = 'nsys_audit' ORDER BY (end - start) DESC LIMIT 1"
        ).fetchone()
        if window:
            totals["window"] = "nsys_audit"
            query = ("SELECT copyKind, COUNT(*), COALESCE(SUM(bytes), 0) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                     "WHERE start >= ? AND end <= ? GROUP BY copyKind")
            rows = connection.execute(query, window).fetchall()
        else:
            rows = connection.execute(
                "SELECT copyKind, COUNT(*), COALESCE(SUM(bytes), 0) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
            ).fetchall()
    for kind, count, total in rows:
        if kind == COPY_KIND_HTOD:
            totals.update(h2d_bytes=total, h2d_count=count)
        elif kind == COPY_KIND_DTOH:
            totals.update(d2h_bytes=total, d2h_count=count)
        else:
            totals.update(other_bytes=totals["other_bytes"] + total, other_count=totals["other_count"] + count)
    return totals


def collect(catalog_path: Path, catalog: dict, models: list[str], precisions: list[str], batches_by_model: dict[str, list[int]],
            nsys_bin: str, repeats: int, run_dir: Path, timeout: int, logger) -> list[dict]:
    script_dir = Path(__file__).resolve().parent
    env = dict(os.environ, LC_ALL="C")
    entries: list[dict] = []
    for precision in precisions:
        for model in models:
            spec = spec_for(catalog, model)
            for batch in batches_by_model.get(model, []):
                stem = f"nsys_{model}_{precision}_b{batch}"
                rep_path = run_dir / "raw" / f"{stem}.nsys-rep"
                sqlite_path = run_dir / "raw" / f"{stem}.sqlite"
                rep_path.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    nsys_bin, "profile", "-t", "cuda,nvtx",
                    "-o", str(run_dir / "raw" / stem), "--force-overwrite", "true",
                    sys.executable, str(script_dir / "forward_target.py"),
                    "--catalog", str(catalog_path), "--model", model, "--batch", str(batch),
                    "--device", "cuda:0", "--precision", precision, "--mode", "nsys",
                    "--nsys-repeats", str(repeats),
                ]
                logger.info("$ " + " ".join(command))
                try:
                    code = subprocess.run(command, env=env, timeout=timeout).returncode
                except subprocess.TimeoutExpired:
                    logger.error(f"  nsys timeout after {timeout}s for {model}/{precision}/B={batch}")
                    code = 124
                expected_h2d = schema_bytes(spec["inputs"]) * batch * repeats
                expected_d2h = schema_bytes(spec["outputs"]) * batch * repeats
                entry = {"model": model, "precision": precision, "batch": batch, "repeats": repeats,
                         "exit_code": code, "expected_h2d_bytes": expected_h2d, "expected_d2h_bytes": expected_d2h,
                         "nsys_rep": str(rep_path)}
                if code == 0 and rep_path.exists():
                    export = [nsys_bin, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(sqlite_path), str(rep_path)]
                    try:
                        subprocess.run(export, env=env, timeout=timeout, check=True, capture_output=True)
                        entry.update(memcpy_totals(sqlite_path))
                        logger.info(f"  nsys {model}/{precision}/B={batch}: H2D {entry['h2d_bytes']:,} B ({entry['h2d_count']} copies), D2H {entry['d2h_bytes']:,} B ({entry['d2h_count']} copies)")
                    except Exception as exc:
                        logger.error(f"  nsys export/parse failed for {model}/{precision}/B={batch}: {exc}")
                entries.append(entry)
    return entries
