#!/usr/bin/env python3
import sys
import os
import re
import glob
from pathlib import Path
import numpy as np

def parse_telemetry_csv(csv_path):
    if not os.path.exists(csv_path):
        return None, None, None, None
    try:
        mem_used = []
        gpu_util = []
        mem_util = []
        with open(csv_path, "r") as f:
            header = f.readline()
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 7:
                    try:
                        g_u = float(parts[3])
                        m_u = float(parts[4])
                        m_used = float(parts[5])
                        gpu_util.append(g_u)
                        mem_util.append(m_u)
                        mem_used.append(m_used)
                    except ValueError:
                        continue
        if not mem_used:
            return None, None, None, None
        
        idle_mb = np.min(mem_used[:max(1, len(mem_used)//10)]) if len(mem_used) > 10 else mem_used[0]
        peak_mb = np.max(mem_used)
        active_mb = peak_mb - idle_mb
        avg_gpu_util = np.mean(gpu_util)
        return idle_mb, peak_mb, active_mb, avg_gpu_util
    except Exception as e:
        print(f"Error parsing telemetry {csv_path}: {e}", file=sys.stderr)
        return None, None, None, None

def parse_solver_log(log_path):
    if not os.path.exists(log_path):
        return None
    text = Path(log_path).read_text(errors="replace")
    
    # Latencies from STEP_TIMING
    step_latencies = []
    for m in re.finditer(r"STEP_TIMING\s+step=(\d+)\s+latency_ms=([0-9.]+)", text):
        step = int(m.group(1))
        ms = float(m.group(2))
        step_latencies.append((step, ms))
    
    if not step_latencies:
        return None
    
    warmup_steps = 2
    steady_latencies = [ms for s, ms in step_latencies if s > warmup_steps]
    if not steady_latencies:
        steady_latencies = [ms for s, ms in step_latencies]
        
    median_ms = float(np.median(steady_latencies))
    mean_ms = float(np.mean(steady_latencies))
    
    # Total samples
    m_samples = re.search(r"total_samples=(\d+)", text)
    total_samples = int(m_samples.group(1)) if m_samples else 2000000
    
    throughput = (total_samples / (median_ms / 1000.0)) if median_ms > 0 else 0.0
    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "throughput": throughput,
        "total_samples": total_samples
    }

def fit_linear_regression(x, y):
    """
    Fits y = V0 + alpha * x
    Returns V0 (MB), alpha_bytes_per_sample, R2
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    if len(x) < 2:
        return float(y[0]), 0.0, 1.0
        
    # Fit y (MB) = V0 + slope_MB_per_sample * x
    p = np.polyfit(x, y, 1)
    slope_mb = p[0]
    v0_mb = p[1]
    
    # Convert slope to bytes/sample: (MB / sample) * 1024 * 1024
    alpha_bytes = slope_mb * 1024.0 * 1024.0
    
    # Compute R^2
    y_pred = np.polyval(p, x)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return v0_mb, alpha_bytes, r2

def main():
    args = sys.argv[1:]
    log_files = []
    if args:
        if len(args) == 1 and os.path.isdir(args[0]):
            log_files = sorted(glob.glob(os.path.join(args[0], "bench_*_gpu_telemetry.csv")))
        else:
            for a in args:
                if os.path.isfile(a):
                    log_files.append(a)
                else:
                    log_files.extend(glob.glob(a))
    else:
        log_files = sorted(glob.glob("logs/bench_*_gpu_telemetry.csv"))

    if not log_files:
        print("No telemetry CSV files found.")
        return

    # Structure: results[provider][batch_size] = {...}
    results = {}
    
    pattern = re.compile(r"bench_([a-zA-Z0-9_]+)_b(\d+)_(.+)_gpu_telemetry\.csv")
    for csv_file in sorted(log_files):
        fname = os.path.basename(csv_file)
        m = pattern.match(fname)
        if not m:
            continue
        provider = m.group(1)
        batch_size = int(m.group(2))
        run_id = m.group(3)
        
        log_dir = os.path.dirname(csv_file)
        solver_log = os.path.join(log_dir, f"bench_{provider}_b{batch_size}_{run_id}.log")
        idle_mb, peak_mb, active_mb, avg_gpu_util = parse_telemetry_csv(csv_file)
        solver_stats = parse_solver_log(solver_log)
        
        if peak_mb is None or solver_stats is None:
            continue
            
        if provider not in results:
            results[provider] = []
            
        results[provider].append({
            "batch_size": batch_size,
            "idle_mb": idle_mb,
            "peak_mb": peak_mb,
            "active_mb": active_mb,
            "gpu_util": avg_gpu_util,
            "median_ms": solver_stats["median_ms"],
            "throughput": solver_stats["throughput"],
            "total_samples": solver_stats["total_samples"]
        })

    print("\n" + "=" * 115)
    print("=== GPU VRAM & THROUGHPUT MEASUREMENTS (H100 SXM5 94GB HBM3, 50MB L2) ===")
    print("=" * 115)
    print(f"{'Provider':<18} {'Batch Size':<12} {'Idle VRAM':<12} {'Peak VRAM':<12} {'Active VRAM':<14} {'Latency(ms)':<14} {'Throughput(M/s)':<16}")
    print("-" * 115)

    regression_data = {}

    for provider, runs in results.items():
        runs.sort(key=lambda r: r["batch_size"])
        batches = [r["batch_size"] for r in runs]
        peaks = [r["peak_mb"] for r in runs]
        
        for r in runs:
            print(f"{provider:<18} {r['batch_size']:<12} {r['idle_mb']:<10.1f} MB {r['peak_mb']:<10.1f} MB {r['active_mb']:<12.1f} MB {r['median_ms']:<12.2f} ms {r['throughput']/1e6:<14.2f}")

        v0_mb, alpha_bytes, r2 = fit_linear_regression(batches, peaks)
        regression_data[provider] = {
            "v0_mb": v0_mb,
            "alpha_bytes": alpha_bytes,
            "r2": r2,
            "runs": runs
        }

    print("\n" + "=" * 115)
    print("=== LINEAR REGRESSION FITTING & ACCELERATOR CAPACITY LIMITS ===")
    print("=" * 115)
    print(f"{'Provider':<18} {'V0 (Static)':<14} {'Alpha (B/sample)':<18} {'R^2 Score':<12} {'Max Batch (94GB)':<18} {'Safe Batch (80GB)':<18}")
    print("-" * 115)

    HBM_MAX_MB = 94000.0   # 94 GB HBM3
    HBM_SAFE_MB = 80000.0  # 80 GB safe operating limit
    L2_CACHE_BYTES = 50.0 * 1024.0 * 1024.0 # 50 MB L2

    for provider, reg in regression_data.items():
        v0 = reg["v0_mb"]
        alpha = reg["alpha_bytes"]
        r2 = reg["r2"]
        
        if alpha > 0.001:
            slope_mb = alpha / (1024.0 * 1024.0)
            max_batch_94gb = int((HBM_MAX_MB - v0) / slope_mb) if HBM_MAX_MB > v0 else 0
            safe_batch_80gb = int((HBM_SAFE_MB - v0) / slope_mb) if HBM_SAFE_MB > v0 else 0
            l2_batch = int(L2_CACHE_BYTES / alpha)
        else:
            max_batch_94gb = int(1e9) # Effectively unlimited by batch parameter (resident tensor limit)
            safe_batch_80gb = int(1e9)
            l2_batch = int(L2_CACHE_BYTES / 72.0) # Raw input size
            
        max_b_str = f"{max_batch_94gb:,}" if max_batch_94gb < 1e8 else "> 100,000,000"
        safe_b_str = f"{safe_batch_80gb:,}" if safe_batch_80gb < 1e8 else "> 80,000,000"
        
        print(f"{provider:<18} {v0:<12.1f} MB {alpha:<16.2f} B {r2:<10.4f} {max_b_str:<18} {safe_b_str:<18}")

    print("\n" + "=" * 115)
    print("=== L2 CACHE (50 MB) WORKING SET ANALYSIS & THROUGHPUT IMPLICATIONS ===")
    print("=" * 115)
    print("Model: WaterCNN flat input ([B, 18] float32 = 72 Bytes/sample, output [B] = 4 Bytes/sample)")
    print(f"Raw Input Working Set @ 50 MB L2 boundary: ~{int(50*1024*1024 / 72):,} samples")
    print(f"Total Working Set (Input + Activations) @ Alpha bytes/sample determines on-chip cache residency.")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
