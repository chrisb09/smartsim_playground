#!/usr/bin/env python3
"""
Hardware Peak Performance and Bandwidth Benchmark for NVIDIA Hopper H100 GPUs.
Measures:
  A) Dense FP32 peak performance (cuBLAS GEMM, allow_tf32=False)
  B) Dense TF32 peak performance (Hopper Tensor Cores, allow_tf32=True, dense)
  C) HBM3 memory bandwidth (out-of-cache D2D copy, triad, read)
  D) NVLink P2P peak data rate (unidirectional and bidirectional for all GPU pairs)

Outputs summary table and structured JSON report.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Please activate the appropriate environment.", file=sys.stderr)
    sys.exit(1)


def get_system_metadata():
    meta = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-set"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", "not-set"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", "not-set"),
        "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT", "not-set"),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }

    # Query nvidia-smi for driver version and GPU details
    try:
        smi_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,pci.bus_id,driver_version,pstate,power.limit,clocks.max.graphics,clocks.max.sm,clocks.max.memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        meta["gpu_devices_smi"] = [line.strip().split(", ") for line in smi_out.stdout.strip().splitlines()]
    except Exception as e:
        meta["gpu_devices_smi_error"] = str(e)

    return meta


def benchmark_dense_gemm(device_idx, tf32_mode, matrix_sizes, warmup_iters=5, timed_iters=20):
    with torch.cuda.device(device_idx):
        torch.backends.cuda.matmul.allow_tf32 = tf32_mode
        torch.backends.cudnn.allow_tf32 = tf32_mode

        mode_name = "TF32" if tf32_mode else "FP32"
        results = []

        for N in matrix_sizes:
            try:
                A = torch.randn(N, N, device=f"cuda:{device_idx}", dtype=torch.float32)
                B = torch.randn(N, N, device=f"cuda:{device_idx}", dtype=torch.float32)

                # Warmup
                for _ in range(warmup_iters):
                    C = torch.mm(A, B)
                torch.cuda.synchronize(device_idx)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                for _ in range(timed_iters):
                    C = torch.mm(A, B)
                end.record()
                torch.cuda.synchronize(device_idx)

                elapsed_ms = start.elapsed_time(end) / timed_iters
                elapsed_s = elapsed_ms / 1000.0

                # GEMM FLOPs = 2 * N^3
                total_flops = 2.0 * (N ** 3)
                tflops = (total_flops / elapsed_s) / 1e12

                results.append({
                    "N": N,
                    "time_ms": elapsed_ms,
                    "tflops": tflops
                })

                del A, B, C
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break

    best = max(results, key=lambda r: r["tflops"]) if results else None
    return {
        "mode": mode_name,
        "allow_tf32": tf32_mode,
        "sweeps": results,
        "peak_tflops": best["tflops"] if best else 0.0,
        "peak_matrix_dim": best["N"] if best else 0
    }


def benchmark_hbm_bandwidth(device_idx, tensor_gb_list=[8, 16, 32], iters=15):
    with torch.cuda.device(device_idx):
        results = []

        for gb in tensor_gb_list:
            num_elements = int((gb * (1024 ** 3)) // 4)
            try:
                x = torch.randn(num_elements, device=f"cuda:{device_idx}", dtype=torch.float32)
                y = torch.empty_like(x)

                # --- 1. D2D Copy (y = x) ---
                # Warmup
                y.copy_(x)
                torch.cuda.synchronize(device_idx)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                for _ in range(iters):
                    y.copy_(x)
                end.record()
                torch.cuda.synchronize(device_idx)

                elapsed_ms = start.elapsed_time(end) / iters
                elapsed_s = elapsed_ms / 1000.0
                # 1 Read + 1 Write = 2 * size
                bytes_transferred = 2.0 * num_elements * 4.0
                tb_s = (bytes_transferred / elapsed_s) / 1e12
                gb_s = (bytes_transferred / elapsed_s) / 1e9

                copy_res = {
                    "tensor_gb": gb,
                    "elements": num_elements,
                    "d2d_copy_time_ms": elapsed_ms,
                    "d2d_copy_tb_s": tb_s,
                    "d2d_copy_gb_s": gb_s
                }

                # --- 2. Read-Only Reduction (sum) ---
                start.record()
                for _ in range(iters):
                    s = torch.sum(x)
                end.record()
                torch.cuda.synchronize(device_idx)

                elapsed_ms_sum = start.elapsed_time(end) / iters
                elapsed_s_sum = elapsed_ms_sum / 1000.0
                bytes_read = float(num_elements * 4.0)
                copy_res["read_time_ms"] = elapsed_ms_sum
                copy_res["read_tb_s"] = (bytes_read / elapsed_s_sum) / 1e12
                copy_res["read_gb_s"] = (bytes_read / elapsed_s_sum) / 1e9

                results.append(copy_res)

                del x, y
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break

    best_copy = max(results, key=lambda r: r["d2d_copy_tb_s"]) if results else None
    return {
        "results": results,
        "peak_copy_tb_s": best_copy["d2d_copy_tb_s"] if best_copy else 0.0,
        "peak_read_tb_s": max([r["read_tb_s"] for r in results]) if results else 0.0,
    }


def benchmark_nvlink_p2p(gpu_pairs, tensor_gb=4.0, iters=25):
    num_elements = int((tensor_gb * (1024 ** 3)) // 4)
    results = []

    for src_idx, dst_idx in gpu_pairs:
        d_src = torch.device(f"cuda:{src_idx}")
        d_dst = torch.device(f"cuda:{dst_idx}")

        can_p2p = torch.cuda.can_device_access_peer(src_idx, dst_idx)
        if not can_p2p:
            results.append({
                "src": src_idx, "dst": dst_idx, "can_p2p": False,
                "unidir_gb_s": 0.0, "bidir_gb_s": 0.0, "status": "P2P unsupported"
            })
            continue

        try:
            with torch.cuda.device(d_src):
                x_src = torch.randn(num_elements, device=d_src, dtype=torch.float32)
            with torch.cuda.device(d_dst):
                y_dst = torch.empty(num_elements, device=d_dst, dtype=torch.float32)

            s_src = torch.cuda.Stream(d_src)
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            # --- Unidirectional Transfer: src -> dst ---
            with torch.cuda.stream(s_src):
                for _ in range(3):
                    y_dst.copy_(x_src)
            torch.cuda.synchronize(d_src)
            torch.cuda.synchronize(d_dst)

            start_evt.record(s_src)
            with torch.cuda.stream(s_src):
                for _ in range(iters):
                    y_dst.copy_(x_src)
            end_evt.record(s_src)
            s_src.synchronize()

            unidir_time_ms = start_evt.elapsed_time(end_evt) / iters
            unidir_gb_s = (num_elements * 4.0 / (unidir_time_ms / 1000.0)) / 1e9

            # --- Bidirectional Transfer: src -> dst AND dst -> src concurrently ---
            with torch.cuda.device(d_dst):
                x_dst = torch.randn(num_elements, device=d_dst, dtype=torch.float32)
            with torch.cuda.device(d_src):
                y_src = torch.empty(num_elements, device=d_src, dtype=torch.float32)

            s_dst = torch.cuda.Stream(d_dst)

            # Warmup
            with torch.cuda.stream(s_src):
                y_dst.copy_(x_src)
            with torch.cuda.stream(s_dst):
                y_src.copy_(x_dst)
            torch.cuda.synchronize(d_src)
            torch.cuda.synchronize(d_dst)

            start_bidir = torch.cuda.Event(enable_timing=True)
            end_bidir = torch.cuda.Event(enable_timing=True)

            start_bidir.record(s_src)
            for _ in range(iters):
                with torch.cuda.stream(s_src):
                    y_dst.copy_(x_src)
                with torch.cuda.stream(s_dst):
                    y_src.copy_(x_dst)
            end_bidir.record(s_src)
            s_dst.synchronize()
            s_src.synchronize()

            bidir_time_ms = start_bidir.elapsed_time(end_bidir) / iters
            # 2 * tensor bytes
            bidir_gb_s = (2.0 * num_elements * 4.0 / (bidir_time_ms / 1000.0)) / 1e9

            results.append({
                "src": src_idx,
                "dst": dst_idx,
                "can_p2p": True,
                "transfer_gb": tensor_gb,
                "unidir_time_ms": unidir_time_ms,
                "unidir_gb_s": unidir_gb_s,
                "bidir_time_ms": bidir_time_ms,
                "bidir_gb_s": bidir_gb_s,
                "status": "OK"
            })

            del x_src, y_dst, x_dst, y_src
            torch.cuda.empty_cache()

        except Exception as e:
            results.append({
                "src": src_idx, "dst": dst_idx, "can_p2p": True,
                "error": str(e), "status": "FAILED"
            })

    return results


def run_bandwidth_test_utility():
    """Runs NVIDIA's demo_suite bandwidthTest if present on system."""
    util_path = "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/extras/demo_suite/bandwidthTest"
    if not os.path.exists(util_path):
        return None
    try:
        res = subprocess.run([util_path, "--memory=pinned", "--mode=quick", "--dtod"],
                             capture_output=True, text=True, timeout=30)
        return res.stdout
    except Exception as e:
        return f"Error executing bandwidthTest: {e}"


def print_summary_tables(fp32_res, tf32_res, hbm_res, nvlink_res, target_gpu):
    print("\n" + "=" * 80)
    print(f" NVIDIA HOPPER H100 HARDWARE BENCHMARK SUMMARY (Target GPU: {target_gpu})")
    print("=" * 80)

    # Compute Table
    print("\n[A & B] DENSE COMPUTE PERFORMANCE (GEMM Sweeps)")
    print("-" * 80)
    print(f"{'Precision / Mode':<28} | {'Best Dimension (N)':<18} | {'Achieved TFLOP/s':<16} | {'Ref. Ceiling':<12}")
    print("-" * 80)
    print(f"{'Dense FP32 (allow_tf32=False)':<28} | N = {fp32_res['peak_matrix_dim']:<14} | {fp32_res['peak_tflops']:>8.2f} TFLOP/s   | ~60-67 TFLOPS")
    print(f"{'Dense TF32 (allow_tf32=True)':<28} | N = {tf32_res['peak_matrix_dim']:<14} | {tf32_res['peak_tflops']:>8.2f} TFLOP/s   | ~482-495 TFLOPS")
    print("-" * 80)
    print("Note: TF32 measurements reflect dense matrix math without 2:4 structured sparsity.")

    # Memory Table
    print("\n[C] HBM3 ON-DEVICE MEMORY BANDWIDTH (Out-of-Cache Sweeps)")
    print("-" * 80)
    print(f"{'Operation':<28} | {'Buffer Size':<18} | {'Bandwidth (GB/s)':<16} | {'Bandwidth (TB/s)'}")
    print("-" * 80)
    for r in hbm_res["results"]:
        print(f"{'D2D Buffer Copy (y = x)':<28} | {r['tensor_gb']} GB ({r['elements']*4/(1024**3):.1f} GiB) | {r['d2d_copy_gb_s']:>8.1f} GB/s     | {r['d2d_copy_tb_s']:.3f} TB/s")
        print(f"{'Read Reduction (sum)':<28} | {r['tensor_gb']} GB ({r['elements']*4/(1024**3):.1f} GiB) | {r['read_gb_s']:>8.1f} GB/s     | {r['read_tb_s']:.3f} TB/s")
    print("-" * 80)
    print(f"Peak Sustained D2D Memory Bandwidth: {hbm_res['peak_copy_tb_s']:.3f} TB/s")

    # NVLink Table
    if nvlink_res:
        print("\n[D] NVLINK PEER-TO-PEER DATA RATE (All Tested GPU Pairs)")
        print("-" * 80)
        print(f"{'Pair':<12} | {'Topology / Type':<18} | {'Unidir (GB/s)':<14} | {'Bidir Agg. (GB/s)':<18} | {'Efficiency (NV6)':<14}")
        print("-" * 80)
        for r in nvlink_res:
            if not r.get("can_p2p", False) or r.get("status") != "OK":
                print(f"GPU {r['src']} <-> {r['dst']:<4} | {'N/A':<18} | {'Unsupported / MIG':<34} | N/A")
                continue

            pair_type = "Intra-Socket" if (r['src'] // 2 == r['dst'] // 2) else "Cross-Socket"
            # Theoretical NV6: 150 GB/s unidir, 300 GB/s bidir
            eff_uni = (r['unidir_gb_s'] / 150.0) * 100.0
            eff_bi = (r['bidir_gb_s'] / 300.0) * 100.0
            print(f"GPU {r['src']} <-> {r['dst']:<4} | {pair_type:<18} | {r['unidir_gb_s']:>6.2f} GB/s     | {r['bidir_gb_s']:>7.2f} GB/s       | {eff_bi:>5.1f}% (Bidir)")
        print("-" * 80)
        print("Reference: 6 bonded NVLinks (NV6) = 150 GB/s unidirectional / 300 GB/s bidirectional peak.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Hopper H100 Hardware Peak Benchmark")
    parser.add_argument("--gpu", type=int, default=0, help="Primary GPU index for compute/HBM benchmarks")
    parser.add_argument("--output", type=str, default="reports/hardware_peaks_report.json", help="Path to save JSON report")
    parser.add_argument("--quick", action="store_true", help="Run quick mode with fewer sweeps")
    parser.add_argument("--skip-nvlink", action="store_true", help="Skip multi-GPU NVLink test")
    parser.add_argument("--nvlink-gpus", type=str, default=None, help="Comma-separated GPU indices for NVLink (e.g. '0,1,3')")
    args = parser.parse_args()

    meta = get_system_metadata()
    print(f"=== Starting H100 Hardware Benchmark on {meta['hostname']} at {meta['timestamp_utc']} ===")
    print(f"Target Primary GPU: {args.gpu} | PyTorch: {meta['pytorch_version']} | CUDA: {meta['cuda_runtime_version']}")

    # Configure sizes
    if args.quick:
        matrix_sizes_fp32 = [4096, 8192]
        matrix_sizes_tf32 = [4096, 8192]
        hbm_sizes = [8, 16]
        iters = 5
    else:
        matrix_sizes_fp32 = [4096, 8192, 12288, 16384, 20480]
        matrix_sizes_tf32 = [4096, 8192, 12288, 16384, 20480, 24576]
        hbm_sizes = [8, 16, 32]
        iters = 20

    # 1. Benchmark Dense FP32
    print("\n>>> Running Target A: Dense FP32 GEMM Sweep...")
    fp32_results = benchmark_dense_gemm(args.gpu, tf32_mode=False, matrix_sizes=matrix_sizes_fp32, timed_iters=iters)
    print(f"    Done. Peak FP32: {fp32_results['peak_tflops']:.2f} TFLOP/s (N={fp32_results['peak_matrix_dim']})")

    # 2. Benchmark Dense TF32
    print("\n>>> Running Target B: Dense TF32 GEMM Sweep...")
    tf32_results = benchmark_dense_gemm(args.gpu, tf32_mode=True, matrix_sizes=matrix_sizes_tf32, timed_iters=iters)
    print(f"    Done. Peak TF32: {tf32_results['peak_tflops']:.2f} TFLOP/s (N={tf32_results['peak_matrix_dim']})")

    # 3. Benchmark HBM3 Bandwidth
    print("\n>>> Running Target C: HBM3 Memory Bandwidth...")
    hbm_results = benchmark_hbm_bandwidth(args.gpu, tensor_gb_list=hbm_sizes, iters=iters)
    print(f"    Done. Peak D2D Copy Bandwidth: {hbm_results['peak_copy_tb_s']:.3f} TB/s")

    # Run auxiliary demo_suite bandwidthTest
    bw_test_output = run_bandwidth_test_utility()

    # 4. Benchmark NVLink P2P
    nvlink_results = []
    if not args.skip_nvlink and torch.cuda.device_count() > 1:
        print("\n>>> Running Target D: NVLink P2P Bandwidth across GPU Pairs...")
        if args.nvlink_gpus:
            gpus = [int(x.strip()) for x in args.nvlink_gpus.split(",") if x.strip().isdigit()]
        else:
            gpus = list(range(torch.cuda.device_count()))

        # Generate all unique pairs
        pairs = [(gpus[i], gpus[j]) for i in range(len(gpus)) for j in range(i + 1, len(gpus))]
        nvlink_results = benchmark_nvlink_p2p(pairs, tensor_gb=4.0, iters=iters)
        print(f"    Done. Tested {len(pairs)} GPU pairs.")

    # Print Summary Table
    print_summary_tables(fp32_results, tf32_results, hbm_results, nvlink_results, args.gpu)

    # Save to JSON
    report = {
        "metadata": meta,
        "target_gpu_idx": args.gpu,
        "dense_fp32": fp32_results,
        "dense_tf32": tf32_results,
        "hbm3_bandwidth": hbm_results,
        "nvlink_p2p": nvlink_results,
        "bandwidth_test_util_stdout": bw_test_output
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full benchmark JSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
