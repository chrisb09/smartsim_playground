#!/usr/bin/env python3
"""
model_inspection/inspect_models.py
====================================
Standalone inspection script for all models in model_catalog.json.

Collects for each model:
  - Artifact file size and SHA-256
  - Parameter count, buffer count, per-dtype element counts (sorted descending)
  - Unique tensor payload bytes (size of the actual weight data stored on disk)
  - Estimated model load memory (RSS delta; reported as observed, not guaranteed)
  - Input/output shapes and per-sample / per-batch byte sizes
  - FLOPs per forward call (per batch entry) via FlopCounterMode on eager reimplementations,
    or analytical estimates for opaque scripted artifacts
  - Peak activation / temporary memory during a forward pass (CPU: process RSS delta;
    CUDA: torch.cuda.max_memory_allocated delta)
  - Steady-state latency (median, mean, p95 over N warmup+measured iterations)
  - Throughput (samples/sec) at one or more batch sizes
  - Coupling operational intensity: FLOPs / (input_bytes_per_entry + output_bytes_per_entry)
  - Device operational intensity: FLOPs / (I/O bytes + weight_bytes/batch) per entry
  - Qualitative classification: compute-bound / balanced / communication-bound

Output:
  - JSON results file (results/<timestamp>_<device>.json)
  - Human-readable Markdown summary (results/<timestamp>_<device>.md)

Usage:
  python inspect_models.py [--device cpu|cuda] [--batch-sizes 1,32,256]
                           [--catalog model_catalog.json] [--warmup 10] [--iters 50]
                           [--timeout-s 30] [--models MODEL_ID[,MODEL_ID,...]]
                           [--skip-timing] [--output-dir results]

Notes on FLOPs:
  FlopCounterMode (torch 2.x) works only on eager (non-scripted) nn.Module subclasses.
  For models where eager reimplementations are available (watercnn, transformer, giant_mlp,
  perfect, mmcp_test_mlp), we instantiate the reimplementation with identical architecture
  and count. For opaque scripted artifacts (mmcp_transformer_core/5input), we use an
  analytical estimate derived from the known architecture and flag it clearly.

Notes on memory:
  - "Weight payload bytes": exact bytes in all parameter + buffer tensors (dtype-aware).
  - "Load RSS delta": RSS before/after torch.jit.load(). Includes interpreter overhead;
    is environment-specific and should be treated as an upper bound.
  - "Forward peak activation bytes" (CPU): RSS delta during a single forward pass in a
    subprocess; approximation only. (CUDA): torch.cuda.max_memory_allocated() delta,
    which is the allocated-peak minus pre-forward allocation, a tighter measure.

Notes on temporary intermediates:
  For a Linear(N, M) layer on batch B, the activation output is B*M*dtype_bytes.
  For N residual blocks of width W, peak live intermediates in fp32 are roughly
  2*B*W*4 bytes (input + output of one block, assuming fused GELU). The exact figure
  depends on PyTorch's allocator and operator fusion. The measured value is always
  preferred over this estimate.

Operational intensity and roofline classification:
  "Coupling OI" = FLOPs / (input_B_per_entry + output_B_per_entry)
    Models the regime where data must be transferred between solver and ML model
    (e.g. over MPI or TCP). Relevant for communication-bound analysis.
  "Device OI"   = FLOPs / (input_B_per_entry + output_B_per_entry + weight_B / batch_size)
    Adds the amortized weight-load cost. Approximates roofline operational intensity
    for a single batch call on a device with no weight caching.
  Both are reported in FLOP/Byte. Reference hardware (H100 NVL 94 GB, our actual cluster GPUs):
    - FP32 (non-TC) peak compute: 60.32 TFLOP/s
    - TF32 Tensor Core peak:     482.6 TFLOP/s (dense) / 965 TFLOP/s (with sparsity)
    - HBM3 peak bandwidth:        3.938 TB/s
    - Ridge point (FP32 non-TC): ~15 FLOP/Byte  (60.32e12 / 3.938e12)
    - Ridge point (TF32 TC):     ~123 FLOP/Byte (482.6e12 / 3.938e12)
    NOTE: PyTorch on H100 uses TF32 by default for matmuls even when inputs are fp32.
    This means fp32-stored models actually run at TF32 throughput in most GEMM kernels.
    Whether a model is memory- or compute-bound therefore depends on which kernel path runs.

  Measured inter-node link bandwidths (from affinity_test/ benchmarks on CLAIX-23):
    - MPI intra-node (same host):  median ~8.83 GiB/s, max ~12.70 GiB/s  (4 MB messages)
    - MPI cross-node:              median ~5.57 GiB/s, max ~ 6.39 GiB/s  (4 MB messages)
    - MPI small-message latency:   median ~0.19 µs (intra-node) .. ~6.46 µs (cross-node)
    - PCIe H2D pinned BW:         median ~18.6 GiB/s, max ~19.5 GiB/s  (16 MB transfers)
    - PCIe D2H pinned BW:         median ~10.3 GiB/s, max ~17.0 GiB/s
  These are the relevant bottleneck bandwidths for coupling_oi comparisons.
  Below the ridge point: memory-bound. Above: compute-bound.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False

# ---------------------------------------------------------------------------
# Constants and hardware reference points
# ---------------------------------------------------------------------------

# H100 NVL 94 GB — the actual GPUs on the CLAIX-23 c23g partition.
# Source: techpowerup.com/gpu-specs/h100-nvl-94-gb.c4327
# Note: "peak compute" figures are theoretical maxima; actual values vary with
# boost/power policy, batch size, and whether TF32 Tensor Cores are engaged.
H100_NVL_TFLOPS_FP32      = 60.32e12   # 60.32 TFLOP/s  (non-TC FP32)
H100_NVL_TFLOPS_TF32      = 482.6e12   # 482.6 TFLOP/s  (TF32 dense, Tensor Core)
H100_NVL_TFLOPS_TF32_SP   = 965.0e12   # 965 TFLOP/s    (TF32 + structured sparsity)
H100_NVL_HBM_BW           = 3.938e12   # 3.938 TB/s HBM3

# PyTorch uses TF32 for matmuls by default on Ampere/Hopper when inputs are fp32.
# The functional precision of fp32 storage with TF32 compute is ~10 bits mantissa.
# Use TF32 ridge point as the primary reference for model forward passes.
H100_NVL_RIDGE_FP32_NONTR = H100_NVL_TFLOPS_FP32 / H100_NVL_HBM_BW   # ~15 FLOP/B
H100_NVL_RIDGE_TF32        = H100_NVL_TFLOPS_TF32 / H100_NVL_HBM_BW   # ~123 FLOP/B

# Default ridge point used for classification: TF32, since that's the default path
# for fp32-stored models under PyTorch on H100.
DEFAULT_RIDGE_POINT = H100_NVL_RIDGE_TF32

# Measured link bandwidths from affinity_test/ benchmarks on CLAIX-23.
# MPI 4 MB messages, 50 iterations, median across all rank pairs.
MEASURED_MPI_INTRA_NODE_BW_GiBS = 8.83    # GiB/s, same-host pairs
MEASURED_MPI_CROSS_NODE_BW_GiBS = 5.57    # GiB/s, cross-host pairs
MEASURED_MPI_INTRA_LATENCY_US   = 0.19    # µs, small (64 B) messages
MEASURED_MPI_CROSS_LATENCY_US   = 6.46    # µs (median all cross-node pairs)
MEASURED_PCIE_H2D_BW_GiBS       = 18.6   # GiB/s, pinned 16 MB
MEASURED_PCIE_D2H_BW_GiBS       = 10.3   # GiB/s, pinned 16 MB

DTYPE_BYTES = {
    "torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2,
    "torch.float64": 8, "torch.int32": 4, "torch.int64": 8,
    "torch.int16": 2, "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1,
}

# ---------------------------------------------------------------------------
# Architecture reimplementations for FlopCounterMode
# ---------------------------------------------------------------------------

class _WaterCNNEager(nn.Module):
    def __init__(self):
        super().__init__()
        self.water_conv   = nn.Conv2d(1,  8,  kernel_size=3, padding=0)
        self.terrain_conv = nn.Conv2d(1,  8,  kernel_size=3, padding=0)
        self.combine_1x1  = nn.Conv2d(2,  8,  kernel_size=1, padding=0)
        self.combine_3x3  = nn.Conv2d(8,  16, kernel_size=3, padding=0)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.relu = nn.ReLU()

    def forward(self, x_water, x_terrain):
        out_water   = self.relu(self.water_conv(x_water))
        out_terrain = self.relu(self.terrain_conv(x_terrain))
        combined    = torch.cat([x_water, x_terrain], dim=1)
        out_comb    = self.relu(self.combine_1x1(combined))
        out_comb    = self.relu(self.combine_3x3(out_comb))
        flat = torch.cat([out_water.flatten(1), out_terrain.flatten(1), out_comb.flatten(1)], dim=1)
        return self.fc(flat).squeeze(1)


class _PerfectModelEager(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("quarter", torch.tensor(0.25, dtype=torch.float32))

    def forward(self, x_water, x_terrain):
        center_w = x_water[:, 0, 1, 1]
        center_t = x_terrain[:, 0, 1, 1]
        this_total = center_w + center_t
        new_value = center_w.clone()
        for nz, nx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            neighbor_total = x_water[:, 0, nz, nx] + x_terrain[:, 0, nz, nx]
            diff = this_total - neighbor_total
            outflow = torch.minimum(center_w, torch.clamp_min(diff, 0.0)) * self.quarter
            inflow  = torch.minimum(x_water[:, 0, nz, nx], torch.clamp_min(-diff, 0.0)) * self.quarter
            new_value = new_value - (outflow - inflow)
        return new_value


class _StableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )
        return self.dropout1(attn_output)


class _WaterTransformerMLPEager(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, ff_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, 9, d_model))
        enc_layer = _StableTransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.global_mlp = nn.Sequential(nn.Linear(18, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU())
        self.head       = nn.Sequential(
            nn.Linear(d_model + 64, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_water, x_terrain):
        b = x_water.shape[0]
        tokens = torch.stack([x_water.squeeze(1), x_terrain.squeeze(1)], dim=-1).view(b, 9, 2)
        tok_feat = self.input_proj(tokens) + self.pos_embed
        tok_feat = self.encoder(tok_feat)
        pooled   = tok_feat.mean(dim=1)
        global_f = self.global_mlp(torch.cat([x_water.flatten(1), x_terrain.flatten(1)], dim=1))
        return self.head(torch.cat([pooled, global_f], dim=1)).squeeze(1)


class _BenchmarkGiantMLPEager(nn.Module):
    def __init__(self, width=4096, depth=12, seed=1337):
        super().__init__()
        rng = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.input_proj = nn.Linear(18, width)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
                for _ in range(depth)
            ])
            self.final_norm = nn.LayerNorm(width)
            self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        finally:
            torch.random.set_rng_state(rng)

    def forward(self, x_water, x_terrain):
        features = torch.cat([x_water.flatten(1), x_terrain.flatten(1)], dim=1)
        hidden = F.gelu(self.input_proj(features))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        hidden = self.final_norm(hidden)
        return self.head(hidden).squeeze(1)


class _MMCPTestMLPEager(nn.Module):
    """
    Test-only MLP matching the MMCP I/O contract.
    Input:  [B, m, 512]  (m history steps of 8^3 = 512 features)
    Output: [B, 2, 512]  (2 forecast steps, matching the MMCP transformer)

    Architecture (shared temporal encoder):
      Per-step projection:  Linear(512, hidden) -> GELU   (applied to each of the m steps)
      Flatten:              [B, m, hidden] -> [B, m*hidden]
      Decoder:              Linear(m*hidden, 2*512) -> reshape to [B, 2, 512]

    Design rationale:
      - Avoids attention entirely; O(m) in time and memory vs O(m^2) for transformers.
      - Significantly smaller and faster than mmcp_transformer while sharing I/O schema.
      - Not intended for training; serves purely as a compute/bandwidth profiling baseline.
    """

    def __init__(self, m: int = 5, hidden: int = 64):
        super().__init__()
        self.m = m
        self.hidden = hidden
        self.step_proj = nn.Linear(512, hidden)
        self.decoder   = nn.Linear(m * hidden, 2 * 512)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, m, 512]
        B, m, _ = src.shape
        # Apply shared step projection to each history step
        h = F.gelu(self.step_proj(src))          # [B, m, hidden]
        h = h.reshape(B, m * self.hidden)        # [B, m*hidden]
        out = self.decoder(h)                    # [B, 2*512]
        return out.reshape(B, 2, 512)            # [B, 2, 512]


# Map model id -> eager reimplementation factory and its input generator
def _make_local_inputs(batch_size: int, device: torch.device):
    """Return (x_water, x_terrain) for local surrogate models."""
    return (
        torch.zeros(batch_size, 1, 3, 3, device=device),
        torch.zeros(batch_size, 1, 3, 3, device=device),
    )

def _make_mmcp_inputs(batch_size: int, m: int, device: torch.device):
    return (torch.zeros(batch_size, m, 512, device=device),)

# Map model id -> eager reimplementation factory and its input generator
def _make_local_inputs(batch_size: int, device: torch.device):
    """Return (x_water, x_terrain) for local surrogate models."""
    return (
        torch.zeros(batch_size, 1, 3, 3, device=device),
        torch.zeros(batch_size, 1, 3, 3, device=device),
    )

def _make_mmcp_inputs(batch_size: int, m: int, device: torch.device):
    return (torch.zeros(batch_size, m, 512, device=device),)


def _perfect_model_flops_per_entry() -> int:
    """
    Manual scalar-operation count for RealFunctionModel (perfect_model).

    FlopCounterMode returns 0 because the model contains no matmuls or
    convolutions — only elementwise ops (add, sub, min, max, mul, clamp, neg).
    These are real floating-point work; we count them manually.

    Per batch entry, per cardinal neighbor (4 neighbors):
      neighbor_total  = x_water[nz,nx] + x_terrain[nz,nx]   : 1 add
      diff            = this_total - neighbor_total           : 1 sub
      clamp(diff, 0)  = max(diff, 0)                         : 1 max
      min(center_w, clamp(diff,0))                           : 1 min
      outflow         = min(...) * quarter                    : 1 mul
      neg_diff        = -diff                                 : 1 neg
      clamp(-diff, 0) = max(-diff, 0)                        : 1 max
      min(x_water[nz,nx], clamp(-diff,0))                   : 1 min
      inflow          = min(...) * quarter                    : 1 mul
      outflow - inflow                                        : 1 sub
      new_value      -= (outflow - inflow)                   : 1 sub
      Subtotal per neighbor: 11 scalar ops
    Pre-loop:
      this_total      = center_w + center_t                  : 1 add
    Total: 4 * 11 + 1 = 45 scalar floating-point ops per entry.

    Note: 'clone()' allocates but performs no arithmetic.
    These are all single-precision scalar ops; no fused-multiply-add path is used.
    """
    return 45


EAGER_REGISTRY: dict[str, tuple] = {
    # model_id -> (factory_fn, input_generator_fn)
    "watercnn":         (lambda: _WaterCNNEager().eval(),                     lambda B, dev: _make_local_inputs(B, dev)),
    "perfect":          (lambda: _PerfectModelEager().eval(),                  lambda B, dev: _make_local_inputs(B, dev)),
    "transformer":      (lambda: _WaterTransformerMLPEager().eval(),           lambda B, dev: _make_local_inputs(B, dev)),
    "giant_mlp":        (lambda: _BenchmarkGiantMLPEager().eval(),             lambda B, dev: _make_local_inputs(B, dev)),
    "mmcp_test_mlp_m5": (lambda: _MMCPTestMLPEager(m=5,  hidden=64).eval(),   lambda B, dev: _make_mmcp_inputs(B, 5,  dev)),
    "mmcp_test_mlp_m10":(lambda: _MMCPTestMLPEager(m=10, hidden=64).eval(),   lambda B, dev: _make_mmcp_inputs(B, 10, dev)),
}

# Analytical FLOPs for opaque MMCP scripted artifacts:
# The MMCP transformer processes [B,5,512] with a standard encoder-decoder transformer.
# Known from the model analysis: ~177.9M learnable params + 5M buffer elements (fp32).
# Architecture: embed(512->d) + 5*pos_emb + N encoder layers (multi-head self-attention +
# FFN) + N decoder layers + output projection. Exact layer dims are not exposed by the
# scripted artifact. We provide a lower-bound estimate assuming:
#   d_model=512, nhead=8, num_encoder_layers~6, num_decoder_layers~6, ff_dim=2048.
# For a single encoder layer with seq_len=5, d=512:
#   QKV projection:    3 * 2 * B * seq_len * d^2     (2 for mul+add)
#   Attention scores:  2 * B * nhead * seq_len^2 * d_head
#   Attn weighted sum: 2 * B * nhead * seq_len^2 * d_head
#   Output proj:       2 * B * seq_len * d^2
#   FFN (2 layers):    2 * 2 * B * seq_len * d * ff_dim
# This estimate is approximate and flagged in results.
def _mmcp_analytical_flops_per_entry(seq_len_src=5, seq_len_tgt=2, d=512, nhead=8,
                                      n_enc=6, n_dec=6, ff_dim=2048) -> int:
    d_head = d // nhead
    # Encoder
    enc_qkv  = 3 * 2 * seq_len_src * d * d        # Q,K,V projections per layer
    enc_attn = 2 * nhead * seq_len_src * seq_len_src * d_head  # scores + weighted sum
    enc_out  = 2 * seq_len_src * d * d             # output proj
    enc_ffn  = 2 * 2 * seq_len_src * d * ff_dim   # two FFN layers
    enc_per_layer = enc_qkv + enc_attn + enc_out + enc_ffn
    # Decoder (cross-attn adds kv_src, self-attn over tgt)
    dec_self_qkv  = 3 * 2 * seq_len_tgt * d * d
    dec_self_attn = 2 * nhead * seq_len_tgt * seq_len_tgt * d_head
    dec_self_out  = 2 * seq_len_tgt * d * d
    dec_cross_q   = 2 * seq_len_tgt * d * d
    dec_cross_kv  = 2 * 2 * seq_len_src * d * d
    dec_cross_attn= 2 * nhead * seq_len_tgt * seq_len_src * d_head
    dec_cross_out = 2 * seq_len_tgt * d * d
    dec_ffn       = 2 * 2 * seq_len_tgt * d * ff_dim
    dec_per_layer = (dec_self_qkv + dec_self_attn + dec_self_out +
                     dec_cross_q + dec_cross_kv + dec_cross_attn + dec_cross_out +
                     dec_ffn)
    # Output projection (d -> 512 per output token; here same as d)
    out_proj = 2 * seq_len_tgt * d * d
    total = n_enc * enc_per_layer + n_dec * dec_per_layer + out_proj
    return int(total)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_proc_rss_bytes(pid: int = None) -> int:
    pid = pid or os.getpid()
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def dtype_byte_size(dtype: torch.dtype) -> int:
    return DTYPE_BYTES.get(str(dtype), 4)


def tensor_payload_bytes(tensors: list) -> int:
    seen = set()
    total = 0
    for t in tensors:
        data_ptr = t.data_ptr()
        if data_ptr not in seen:
            seen.add(data_ptr)
            total += t.numel() * dtype_byte_size(t.dtype)
    return total


def collect_param_dtype_counts(model: nn.Module) -> dict[str, dict[str, int]]:
    """Return {'parameters': {dtype_str: elem_count, ...}, 'buffers': {...}}."""
    params_dtypes: dict[str, int] = collections.Counter()
    bufs_dtypes:   dict[str, int] = collections.Counter()
    for p in model.parameters():
        params_dtypes[str(p.dtype)] += p.numel()
    for b in model.buffers():
        bufs_dtypes[str(b.dtype)] += b.numel()
    # Sort by count descending
    return {
        "parameters": dict(sorted(params_dtypes.items(), key=lambda x: -x[1])),
        "buffers":    dict(sorted(bufs_dtypes.items(),   key=lambda x: -x[1])),
    }


def io_bytes_per_entry(schema_inputs: list, schema_outputs: list) -> tuple[int, int]:
    """Compute per-sample input and output bytes from catalog schema."""
    def _elem_bytes(shape: list, dtype: str) -> int:
        elems = 1
        for d in shape:
            if d is not None:
                elems *= d
        return elems * DTYPE_BYTES.get(dtype, 4)

    in_bytes  = sum(_elem_bytes(inp["shape"][1:], inp["dtype"]) for inp in schema_inputs)
    out_bytes = sum(_elem_bytes(out["shape"][1:], out["dtype"]) for out in schema_outputs)
    return in_bytes, out_bytes


def measure_flops_eager(model_id: str, batch_size: int, device: torch.device) -> dict:
    """Run FlopCounterMode on eager reimplementation; returns flop counts or error.

    Special case: perfect_model (RealFunctionModel) uses only elementwise ops that
    FlopCounterMode cannot instrument (min, max, clamp, add, sub, mul on scalars).
    For this model we fall back to a manual count via _perfect_model_flops_per_entry().
    """
    if not HAS_FLOP_COUNTER:
        return {"total_flops": None, "error": "FlopCounterMode not available"}
    if model_id not in EAGER_REGISTRY:
        return {"total_flops": None, "error": f"No eager reimplementation for {model_id}"}

    # perfect_model: FlopCounterMode returns 0; use manual elementwise-op count instead.
    if model_id == "perfect":
        flops_per_entry = _perfect_model_flops_per_entry()
        return {
            "total_flops": flops_per_entry * batch_size,
            "flops_per_entry": flops_per_entry,
            "method": "manual_elementwise_count",
            "batch_size_used": batch_size,
            "op_breakdown": {
                "add/sub (scalar)": 5 * 4 + 1,   # 4 neighbors * 5 + pre-loop
                "min/max/clamp (scalar)": 3 * 4,
                "mul (scalar)": 2 * 4,
                "neg (scalar)": 1 * 4,
            },
            "note": (
                "FlopCounterMode returns 0 for elementwise ops (min/max/clamp/add/sub/mul). "
                "Count is manually derived from the solver update loop; see _perfect_model_flops_per_entry()."
            ),
        }

    factory_fn, input_fn = EAGER_REGISTRY[model_id]
    model  = factory_fn()
    inputs = input_fn(batch_size, device)

    try:
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            _ = model(*inputs)
        total = flop_counter.get_total_flops()
        counts = flop_counter.get_flop_counts()
        # Flatten op counts from top-level "Global" key
        global_ops = counts.get("Global", {})
        op_breakdown = {str(k): v for k, v in global_ops.items()}
        return {
            "total_flops": total,
            "flops_per_entry": total // batch_size if batch_size else total,
            "op_breakdown": op_breakdown,
            "method": "FlopCounterMode_eager",
            "batch_size_used": batch_size,
        }
    except Exception as exc:
        return {"total_flops": None, "error": str(exc)}


def measure_flops_analytical(model_id: str, batch_size: int) -> dict:
    """Analytical FLOPs estimate for opaque artifacts."""
    if model_id in ("mmcp_transformer_core", "mmcp_transformer_5input"):
        flops_per_entry = _mmcp_analytical_flops_per_entry()
        return {
            "total_flops": flops_per_entry * batch_size,
            "flops_per_entry": flops_per_entry,
            "method": "analytical_estimate_lower_bound",
            "batch_size_used": batch_size,
            "warning": (
                "Estimate assumes d_model=512, nhead=8, n_enc=6, n_dec=6, ff_dim=2048. "
                "Actual architecture is opaque. This is a lower bound."
            ),
        }
    return {"total_flops": None, "error": f"No analytical estimate for {model_id}"}


def measure_activation_memory_cpu(model: nn.Module, inputs: tuple) -> dict:
    """
    Measure peak RSS increase during a single forward pass on CPU.
    Uses a subprocess trick to get a clean baseline; falls back to in-process delta.
    """
    gc.collect()
    rss_before = get_proc_rss_bytes()
    with torch.inference_mode():
        _ = model(*inputs)
    gc.collect()
    rss_after = get_proc_rss_bytes()
    delta = max(0, rss_after - rss_before)
    return {
        "peak_forward_bytes": delta,
        "method": "proc_rss_delta_cpu",
        "note": "RSS delta during forward; includes allocator overhead, not just live activations.",
    }


def measure_activation_memory_cuda(model: nn.Module, inputs: tuple) -> dict:
    """Measure peak CUDA allocation increase during a single forward pass."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    alloc_before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        _ = model(*inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    alloc_after = torch.cuda.memory_allocated()
    return {
        "peak_forward_bytes": max(0, peak - alloc_before),
        "alloc_delta_bytes": max(0, alloc_after - alloc_before),
        "method": "cuda_max_memory_allocated_delta",
        "note": "Peak allocated minus pre-forward allocated. Includes live activations + workspace.",
    }


def time_forward(model, inputs: tuple, device: torch.device,
                 warmup: int, iters: int, timeout_s: float,
                 flops_per_entry: Optional[int] = None) -> dict:
    """Time model.forward over warmup+iters runs. Returns latency statistics.

    If flops_per_entry is provided, also computes achieved_flops_per_s
    (= flops_per_entry * batch_size / median_latency_s), the y-axis value
    for a roofline plot.
    """
    is_cuda = device.type == "cuda"

    # Warmup
    try:
        for _ in range(warmup):
            with torch.inference_mode():
                _ = model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
    except Exception as exc:
        return {"error": f"Warmup failed: {exc}"}

    # Timed runs
    latencies_s = []
    deadline = time.monotonic() + timeout_s
    for _ in range(iters):
        if time.monotonic() > deadline:
            break
        try:
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event   = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.inference_mode():
                    _ = model(*inputs)
                end_event.record()
                torch.cuda.synchronize()
                latencies_s.append(start_event.elapsed_time(end_event) / 1000.0)
            else:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = model(*inputs)
                latencies_s.append(time.perf_counter() - t0)
        except Exception as exc:
            return {"error": f"Forward failed at iter {len(latencies_s)}: {exc}"}

    if not latencies_s:
        return {"error": "No measurements completed within timeout"}

    batch_size = inputs[0].shape[0]
    med_s   = statistics.median(latencies_s)
    mean_s  = statistics.mean(latencies_s)
    p95_s   = sorted(latencies_s)[int(0.95 * len(latencies_s))]
    min_s   = min(latencies_s)
    max_s   = max(latencies_s)

    return {
        "n_measured":         len(latencies_s),
        "batch_size":         batch_size,
        "median_latency_s":   med_s,
        "mean_latency_s":     mean_s,
        "p95_latency_s":      p95_s,
        "min_latency_s":      min_s,
        "max_latency_s":      max_s,
        "throughput_samples_per_s": batch_size / med_s,
        # Roofline y-axis: achieved compute throughput.
        # = FLOPs executed per forward call / wall time for that call.
        # Only set when flops_per_entry is known; None otherwise.
        "achieved_flops_per_s": (
            flops_per_entry * batch_size / med_s
            if flops_per_entry is not None else None
        ),
        "achieved_gflops_per_s": (
            flops_per_entry * batch_size / med_s / 1e9
            if flops_per_entry is not None else None
        ),
    }


def classify_regime(coupling_oi: Optional[float], device_oi: Optional[float],
                    ridge_point: float = DEFAULT_RIDGE_POINT) -> dict:
    """
    Classify model into compute/balanced/memory/communication regime.

    coupling_oi: FLOPs / (in_bytes + out_bytes) per entry.
      Compares compute to the coupling-link data cost (MPI/TCP).
      Relevant BW references (measured, CLAIX-23):
        MPI intra-node ~8.8 GiB/s, MPI cross-node ~5.6 GiB/s.
    device_oi:   FLOPs / (in+out+weight/B) per entry.
      Compares compute to the on-device HBM data cost (amortised over batch).
      Relevant BW reference: H100 NVL HBM3 ~3.94 TB/s.

    Ridge points (H100 NVL 94 GB, single card):
      FP32 non-TC:   ~15 FLOP/B  (60.32 TFLOP/s / 3.938 TB/s)
      TF32 TC dense: ~123 FLOP/B (482.6 TFLOP/s / 3.938 TB/s)
    PyTorch uses TF32 for fp32 matmuls by default on Hopper; DEFAULT_RIDGE_POINT = TF32 ridge.
    """
    def _classify(oi, rp):
        if oi is None:
            return "unknown (FLOPs not available)"
        if oi < 1.0:
            return "communication-bound (OI < 1 FLOP/Byte; dominated by data-transfer latency)"
        elif oi < rp * 0.1:
            return f"memory-bound (OI {oi:.1f} << ridge {rp:.0f} FLOP/B)"
        elif oi < rp * 0.5:
            return f"memory-bound (OI {oi:.1f} < ridge {rp:.0f} FLOP/B)"
        elif oi < rp * 2.0:
            return f"near ridge / balanced (OI {oi:.1f} ~ ridge {rp:.0f} FLOP/B)"
        else:
            return f"compute-bound (OI {oi:.1f} >> ridge {rp:.0f} FLOP/B)"

    return {
        "coupling_regime": _classify(coupling_oi, ridge_point),
        "device_regime":   _classify(device_oi,   ridge_point),
        "ridge_point_used_flop_per_byte": ridge_point,
        "ridge_point_label": "H100 NVL TF32 dense TC (default for fp32 PyTorch matmuls)",
        "fp32_nonTC_ridge": round(H100_NVL_RIDGE_FP32_NONTR, 1),
        "tf32_TC_ridge":    round(H100_NVL_RIDGE_TF32, 1),
    }


# ---------------------------------------------------------------------------
# Main inspection logic per model
# ---------------------------------------------------------------------------

def resolve_path(catalog_dir: Path, rel_path: Optional[str]) -> Optional[str]:
    if rel_path is None:
        return None
    resolved = (catalog_dir / rel_path).resolve()
    return str(resolved) if resolved.exists() else None


def inspect_one_model(entry: dict, catalog_dir: Path, device: torch.device,
                      batch_sizes: list[int], warmup: int, iters: int,
                      timeout_s: float, skip_timing: bool) -> dict:
    model_id  = entry["id"]
    flops_mode = entry.get("flops_mode", "eager_reimplementation")

    # Resolve artifact path
    device_key = "artifact_cuda" if device.type == "cuda" else "artifact_cpu"
    artifact_rel = entry.get(device_key) or entry.get("artifact_cpu")
    artifact_path = resolve_path(catalog_dir, artifact_rel)

    result: dict = {
        "model_id":    model_id,
        "display_name": entry.get("display_name", model_id),
        "family":      entry.get("family", "unknown"),
        "device":      str(device),
        "artifact":    {},
        "parameters":  {},
        "io":          {},
        "flops":       {},
        "memory":      {},
        "timing":      {},
        "metrics":     {},
        "notes":       entry.get("notes", ""),
        "errors":      [],
    }

    # ------------------------------------------------------------------
    # 1. Artifact file stats
    # ------------------------------------------------------------------
    if artifact_path is None:
        result["errors"].append(f"Artifact not found at {artifact_rel}")
        result["artifact"] = {"path": artifact_rel, "exists": False}
    else:
        try:
            stat_result = os.stat(artifact_path)
            result["artifact"] = {
                "path":          artifact_path,
                "exists":        True,
                "size_bytes":    stat_result.st_size,
                "size_mib":      round(stat_result.st_size / (1024**2), 4),
                "sha256":        sha256_file(artifact_path),
            }
        except Exception as exc:
            result["errors"].append(f"File stat/hash failed: {exc}")
            result["artifact"] = {"path": artifact_path, "exists": True, "size_bytes": None}

    # ------------------------------------------------------------------
    # 2. Load model and collect parameter/buffer statistics
    # ------------------------------------------------------------------
    model = None
    if artifact_path:
        gc.collect()
        rss_before_load = get_proc_rss_bytes()
        cuda_before_load = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        try:
            model = torch.jit.load(artifact_path, map_location=device)
            model.eval()
        except Exception as exc:
            result["errors"].append(f"Model load failed: {exc}")
        rss_after_load  = get_proc_rss_bytes()
        cuda_after_load = torch.cuda.memory_allocated() if device.type == "cuda" else 0

        result["memory"]["load_rss_delta_bytes"]  = max(0, rss_after_load - rss_before_load)
        result["memory"]["load_rss_delta_mib"]    = round(max(0, rss_after_load - rss_before_load) / (1024**2), 4)
        if device.type == "cuda":
            result["memory"]["load_cuda_delta_bytes"] = max(0, cuda_after_load - cuda_before_load)
            result["memory"]["load_cuda_delta_mib"]   = round(max(0, cuda_after_load - cuda_before_load) / (1024**2), 4)

    if model is not None:
        try:
            all_params  = list(model.parameters())
            all_buffers = list(model.buffers())
            param_count = sum(p.numel() for p in all_params)
            buf_count   = sum(b.numel() for b in all_buffers)

            param_payload  = tensor_payload_bytes(all_params)
            buf_payload    = tensor_payload_bytes(all_buffers)
            total_payload  = param_payload + buf_payload

            dtype_counts = collect_param_dtype_counts(model)

            result["parameters"] = {
                "trainable_param_count":      param_count,
                "buffer_elem_count":          buf_count,
                "total_tensor_elem_count":    param_count + buf_count,
                "param_payload_bytes":        param_payload,
                "param_payload_mib":          round(param_payload / (1024**2), 4),
                "buffer_payload_bytes":       buf_payload,
                "buffer_payload_mib":         round(buf_payload / (1024**2), 4),
                "total_payload_bytes":        total_payload,
                "total_payload_mib":          round(total_payload / (1024**2), 4),
                "dtype_counts_params":        dtype_counts["parameters"],
                "dtype_counts_buffers":       dtype_counts["buffers"],
                "dtype_note": (
                    "Element counts per dtype, sorted descending. "
                    "Payload bytes computed from unique tensor data pointers to avoid "
                    "double-counting shared/aliased tensors."
                ),
            }
        except Exception as exc:
            result["errors"].append(f"Parameter inspection failed: {exc}")

    # ------------------------------------------------------------------
    # 3. I/O schema
    # ------------------------------------------------------------------
    schema_inputs  = entry.get("inputs", [])
    schema_outputs = entry.get("outputs", [])
    in_bytes, out_bytes = io_bytes_per_entry(schema_inputs, schema_outputs)
    result["io"] = {
        "inputs":              schema_inputs,
        "outputs":             schema_outputs,
        "input_bytes_per_entry":  in_bytes,
        "output_bytes_per_entry": out_bytes,
        "total_io_bytes_per_entry": in_bytes + out_bytes,
        "io_ratio_out_over_in": round(out_bytes / in_bytes, 4) if in_bytes > 0 else None,
    }

    # ------------------------------------------------------------------
    # 4. FLOPs
    # ------------------------------------------------------------------
    # Always measure at batch_size=1 for per-entry FLOPs
    flops_result: dict = {}
    if flops_mode == "eager_reimplementation":
        flops_result = measure_flops_eager(model_id, batch_size=1, device=torch.device("cpu"))
    elif flops_mode == "analytical_estimate":
        flops_result = measure_flops_analytical(model_id, batch_size=1)
    else:
        flops_result = {"total_flops": None, "error": f"Unknown flops_mode: {flops_mode}"}

    result["flops"] = flops_result
    flops_per_entry = flops_result.get("flops_per_entry") or flops_result.get("total_flops")

    # ------------------------------------------------------------------
    # 5. Derived metrics (operational intensity)
    # ------------------------------------------------------------------
    weight_bytes = result["parameters"].get("total_payload_bytes", 0)
    coupling_oi  = None
    device_oi_b1 = None
    io_total     = in_bytes + out_bytes

    if flops_per_entry is not None and io_total > 0:
        coupling_oi = flops_per_entry / io_total
    if flops_per_entry is not None and io_total > 0:
        # Device OI at batch=1: weights must be loaded once per call
        device_oi_b1 = flops_per_entry / (io_total + weight_bytes)

    regime = classify_regime(coupling_oi, device_oi_b1)

    result["metrics"] = {
        "coupling_oi_flop_per_byte":       round(coupling_oi, 4) if coupling_oi is not None else None,
        "device_oi_batch1_flop_per_byte":  round(device_oi_b1, 4) if device_oi_b1 is not None else None,
        "weight_bytes":                    weight_bytes,
        "in_bytes_per_entry":              in_bytes,
        "out_bytes_per_entry":             out_bytes,
        "flops_per_entry":                 flops_per_entry,
        "regime":                          regime,
        "note": (
            "coupling_oi = FLOPs / (in+out bytes per entry). "
            "Models the controller-to-GPU data-transfer bottleneck (MPI/TCP). "
            "device_oi (batch=1) = FLOPs / (in+out+weights bytes). "
            "Increasing batch size amortises weights: device_oi -> coupling_oi as B->inf. "
            "H100 NVL 94 GB reference: FP32 non-TC ridge ~15 FLOP/B, TF32 TC ridge ~123 FLOP/B. "
            "PyTorch fp32 matmuls use TF32 TC by default on Hopper; classification uses TF32 ridge. "
            f"Measured link BW: MPI intra-node ~{MEASURED_MPI_INTRA_NODE_BW_GiBS} GiB/s, "
            f"MPI cross-node ~{MEASURED_MPI_CROSS_NODE_BW_GiBS} GiB/s, "
            f"PCIe H2D ~{MEASURED_PCIE_H2D_BW_GiBS} GiB/s (from affinity_test/ benchmarks)."
        ),
    }

    # Device OI at each batch size
    device_oi_by_batch = {}
    if flops_per_entry is not None and io_total > 0:
        for bs in batch_sizes:
            total_io = bs * io_total
            total_weight = weight_bytes
            device_oi_by_batch[bs] = round(
                (flops_per_entry * bs) / (total_io + total_weight), 4
            )
    result["metrics"]["device_oi_by_batch"] = device_oi_by_batch

    # ------------------------------------------------------------------
    # 6. Timing & activation memory
    # ------------------------------------------------------------------
    if not skip_timing and model is not None:
        timing_by_batch = {}
        activation_by_batch = {}

        for bs in batch_sizes:
            # Build scripted-model inputs from schema
            try:
                raw_inputs = _build_scripted_inputs(entry, bs, device)
            except Exception as exc:
                timing_by_batch[bs] = {"error": f"Could not build inputs: {exc}"}
                continue

            # Activation memory at this batch size
            try:
                if device.type == "cuda":
                    act_mem = measure_activation_memory_cuda(model, raw_inputs)
                else:
                    act_mem = measure_activation_memory_cpu(model, raw_inputs)
                activation_by_batch[bs] = act_mem
            except Exception as exc:
                activation_by_batch[bs] = {"error": str(exc)}

            # Timing
            lat = time_forward(model, raw_inputs, device, warmup, iters, timeout_s,
                               flops_per_entry=flops_per_entry)
            timing_by_batch[bs] = lat

        result["timing"]  = timing_by_batch
        result["memory"]["activation_by_batch"] = activation_by_batch
    else:
        result["timing"] = {"skipped": True}

    return result


def _build_scripted_inputs(entry: dict, batch_size: int, device: torch.device) -> tuple:
    """Build zero-valued input tensors from catalog schema for a scripted model."""
    inputs = []
    for inp in entry["inputs"]:
        shape = [batch_size if d is None else d for d in inp["shape"]]
        dtype_map = {
            "float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16, "int32": torch.int32, "int64": torch.int64,
        }
        dtype = dtype_map.get(inp["dtype"], torch.float32)
        inputs.append(torch.zeros(shape, dtype=dtype, device=device))
    return tuple(inputs)


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _fmt_bytes(b: Optional[int]) -> str:
    if b is None:
        return "N/A"
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.2f} KiB"
    elif b < 1024**3:
        return f"{b/1024**2:.2f} MiB"
    else:
        return f"{b/1024**3:.3f} GiB"


def _fmt_flops(f: Optional[int]) -> str:
    if f is None:
        return "N/A"
    if f < 1e3:
        return f"{f} FLOP"
    elif f < 1e6:
        return f"{f/1e3:.2f} kFLOP"
    elif f < 1e9:
        return f"{f/1e6:.2f} MFLOP"
    elif f < 1e12:
        return f"{f/1e9:.3f} GFLOP"
    else:
        return f"{f/1e12:.3f} TFLOP"


def _fmt_lat(s: Optional[float]) -> str:
    if s is None:
        return "N/A"
    if s < 1e-3:
        return f"{s*1e6:.1f} µs"
    elif s < 1.0:
        return f"{s*1e3:.2f} ms"
    else:
        return f"{s:.3f} s"


def _fmt_throughput(samples_per_s: Optional[float]) -> str:
    if samples_per_s is None:
        return "N/A"
    if samples_per_s >= 1e6:
        return f"{samples_per_s/1e6:.2f}M samp/s"
    elif samples_per_s >= 1e3:
        return f"{samples_per_s/1e3:.1f}k samp/s"
    else:
        return f"{samples_per_s:.1f} samp/s"


def generate_markdown(all_results: list[dict], run_meta: dict, batch_sizes: list[int]) -> str:
    lines = []
    lines.append("# Model Inspection Report")
    lines.append("")
    lines.append(f"**Generated:** {run_meta.get('timestamp', 'unknown')}  ")
    lines.append(f"**Device:** {run_meta.get('device', 'unknown')}  ")
    if run_meta.get("cuda_device_name"):
        lines.append(f"**GPU:** {run_meta['cuda_device_name']} ({run_meta.get('cuda_total_memory_gib', '?')} GiB)  ")
    lines.append(f"**PyTorch:** {run_meta.get('torch_version', '?')}  ")
    lines.append(f"**Python:** {run_meta.get('python_version', '?')}  ")
    lines.append(f"**Hostname:** {run_meta.get('hostname', '?')}  ")
    lines.append("")

    # ---- Summary table ----
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Params | Buffers | Payload | Artifact | FLOPs/entry | Coupling OI | Device OI (B=1) |")
    lines.append("|-------|--------|---------|---------|----------|-------------|-------------|-----------------|")
    for r in all_results:
        p = r.get("parameters", {})
        m = r.get("metrics", {})
        a = r.get("artifact", {})
        f = r.get("flops", {})
        param_count = p.get("trainable_param_count", "?")
        buf_count   = p.get("buffer_elem_count", "?")
        payload     = _fmt_bytes(p.get("total_payload_bytes"))
        art_size    = _fmt_bytes(a.get("size_bytes"))
        flops_entry = _fmt_flops(m.get("flops_per_entry"))
        coup_oi     = m.get("coupling_oi_flop_per_byte")
        dev_oi      = m.get("device_oi_batch1_flop_per_byte")
        coup_str    = f"{coup_oi:.2f} FLOP/B" if coup_oi is not None else "N/A"
        dev_str     = f"{dev_oi:.4f} FLOP/B" if dev_oi is not None else "N/A"
        flops_method = f.get("method", "")
        if "analytical" in flops_method:
            flops_entry += " *"
        lines.append(f"| {r['display_name']} | {param_count:,} | {buf_count:,} | {payload} | {art_size} | {flops_entry} | {coup_str} | {dev_str} |")

    lines.append("")
    lines.append("*\\* Analytical estimate (opaque scripted artifact; lower bound)*")
    lines.append("")

    # ---- I/O table ----
    lines.append("## I/O Shapes and Per-Entry Byte Cost")
    lines.append("")
    lines.append("| Model | Inputs | Outputs | In B/entry | Out B/entry | Total I/O B/entry | Out/In ratio |")
    lines.append("|-------|--------|---------|-----------|------------|-------------------|-------------|")
    for r in all_results:
        io = r.get("io", {})
        in_shapes  = "; ".join(f"{i['name']}:{i['shape']}" for i in io.get("inputs", []))
        out_shapes = "; ".join(f"{o['name']}:{o['shape']}" for o in io.get("outputs", []))
        in_b  = _fmt_bytes(io.get("input_bytes_per_entry"))
        out_b = _fmt_bytes(io.get("output_bytes_per_entry"))
        tot_b = _fmt_bytes(io.get("total_io_bytes_per_entry"))
        ratio = io.get("io_ratio_out_over_in")
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        lines.append(f"| {r['display_name']} | {in_shapes} | {out_shapes} | {in_b} | {out_b} | {tot_b} | {ratio_str} |")
    lines.append("")

    # ---- Parameter dtypes ----
    lines.append("## Parameter and Buffer Dtype Distribution")
    lines.append("")
    for r in all_results:
        p = r.get("parameters", {})
        lines.append(f"### {r['display_name']}")
        pd = p.get("dtype_counts_params", {})
        bd = p.get("dtype_counts_buffers", {})
        if pd:
            lines.append("**Parameters (trainable):**")
            for dtype, count in pd.items():
                lines.append(f"  - `{dtype}`: {count:,} elements ({_fmt_bytes(count * DTYPE_BYTES.get(dtype, 4))})")
        else:
            lines.append("**Parameters:** none")
        if bd:
            lines.append("**Buffers:**")
            for dtype, count in bd.items():
                lines.append(f"  - `{dtype}`: {count:,} elements ({_fmt_bytes(count * DTYPE_BYTES.get(dtype, 4))})")
        else:
            lines.append("**Buffers:** none")
        lines.append("")

    # ---- Memory ----
    lines.append("## Memory Costs")
    lines.append("")
    lines.append("### Model Load Cost")
    lines.append("")
    lines.append("| Model | Payload | RSS delta (load) | CUDA delta (load) |")
    lines.append("|-------|---------|-----------------|-------------------|")
    for r in all_results:
        p = r.get("parameters", {})
        mem = r.get("memory", {})
        payload  = _fmt_bytes(p.get("total_payload_bytes"))
        rss_load = _fmt_bytes(mem.get("load_rss_delta_bytes"))
        cuda_load = _fmt_bytes(mem.get("load_cuda_delta_bytes")) if "load_cuda_delta_bytes" in mem else "CPU-only"
        lines.append(f"| {r['display_name']} | {payload} | {rss_load} | {cuda_load} |")
    lines.append("")
    lines.append("*RSS delta includes Python/JIT interpreter overhead. CUDA delta is more precise.*")
    lines.append("")

    lines.append("### Forward Pass Peak Activation/Temporary Memory")
    lines.append("")
    header_bss = " | ".join(f"B={bs}" for bs in batch_sizes)
    lines.append(f"| Model | {header_bss} |")
    lines.append("|-------|" + "|".join(["---"] * len(batch_sizes)) + "|")
    for r in all_results:
        mem = r.get("memory", {})
        act = mem.get("activation_by_batch", {})
        row = [r['display_name']]
        for bs in batch_sizes:
            a = act.get(bs, {})
            if "error" in a:
                row.append(f"err: {a['error'][:30]}")
            elif "peak_forward_bytes" in a:
                row.append(_fmt_bytes(a["peak_forward_bytes"]))
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- Timing ----
    lines.append("## Timing (Latency and Throughput)")
    lines.append("")
    for bs in batch_sizes:
        lines.append(f"### Batch size = {bs}")
        lines.append("")
        lines.append("| Model | Median lat | Mean lat | P95 lat | Throughput | Achieved GFLOP/s |")
        lines.append("|-------|-----------|---------|--------|-----------|-----------------|")
        for r in all_results:
            timing = r.get("timing", {})
            if timing.get("skipped"):
                row_str = "skipped | skipped | skipped | skipped | skipped"
            else:
                t = timing.get(bs, {})
                if "error" in t:
                    row_str = f"err: {t['error'][:40]} | | | | "
                else:
                    gflops = t.get("achieved_gflops_per_s")
                    gflops_str = f"{gflops:.2f}" if gflops is not None else "N/A"
                    row_str = (
                        f"{_fmt_lat(t.get('median_latency_s'))} | "
                        f"{_fmt_lat(t.get('mean_latency_s'))} | "
                        f"{_fmt_lat(t.get('p95_latency_s'))} | "
                        f"{_fmt_throughput(t.get('throughput_samples_per_s'))} | "
                        f"{gflops_str}"
                    )
            lines.append(f"| {r['display_name']} | {row_str} |")
        lines.append("")

    # ---- Operational Intensity ----
    lines.append("## Operational Intensity and Roofline Analysis")
    lines.append("")
    lines.append(
        "**Coupling OI** = FLOPs / (input bytes + output bytes) per entry.  \n"
        "  Models the data-transfer bottleneck for coupling the solver to the ML model (MPI/TCP).  \n"
        "**Device OI** = FLOPs / (input+output+weights/batch) per entry.  \n"
        "  Adds amortised weight-load cost; approaches Coupling OI as batch size → ∞.  \n"
        "\n"
        "**H100 NVL 94 GB reference (our actual cluster GPUs, CLAIX-23 c23g partition):**  \n"
        f"  - FP32 non-TC peak: {H100_NVL_TFLOPS_FP32/1e12:.2f} TFLOP/s  "
        f"  HBM3 BW: {H100_NVL_HBM_BW/1e12:.3f} TB/s  \n"
        f"  - TF32 TC dense peak: {H100_NVL_TFLOPS_TF32/1e12:.1f} TFLOP/s  "
        f"  (PyTorch default for fp32 matmuls on Hopper)  \n"
        f"  - Ridge point — FP32 non-TC: **~{H100_NVL_RIDGE_FP32_NONTR:.0f} FLOP/B**  |  "
        f"TF32 TC: **~{H100_NVL_RIDGE_TF32:.0f} FLOP/B**  \n"
        "\n"
        "**Measured link bandwidths (affinity_test/, CLAIX-23):**  \n"
        f"  - MPI intra-node (4 MB): median **{MEASURED_MPI_INTRA_NODE_BW_GiBS} GiB/s**  "
        f"  latency: {MEASURED_MPI_INTRA_LATENCY_US} µs  \n"
        f"  - MPI cross-node  (4 MB): median **{MEASURED_MPI_CROSS_NODE_BW_GiBS} GiB/s**  "
        f"  latency: {MEASURED_MPI_CROSS_LATENCY_US} µs  \n"
        f"  - PCIe H2D pinned (16 MB): median **{MEASURED_PCIE_H2D_BW_GiBS} GiB/s**  "
        f"  D2H: {MEASURED_PCIE_D2H_BW_GiBS} GiB/s  \n"
        "\n"
        "*(Small models may fit in GPU L2/L3 cache after the first call, raising effective "
        "OI beyond the HBM roofline. Exact position requires hardware performance counters "
        "(e.g. NCU) to measure realized memory traffic.)*"
    )
    lines.append("")
    lines.append("| Model | Coupling OI | " + " | ".join(f"Device OI B={bs}" for bs in batch_sizes) + " | Coupling regime | Device regime (B=1) |")
    lines.append("|-------|------------|" + "|".join(["---"] * len(batch_sizes)) + "|---|---|")
    for r in all_results:
        m = r.get("metrics", {})
        coup_oi = m.get("coupling_oi_flop_per_byte")
        coup_str = f"{coup_oi:.2f}" if coup_oi is not None else "N/A"
        dev_oi_by = m.get("device_oi_by_batch", {})
        dev_strs = []
        for bs in batch_sizes:
            v = dev_oi_by.get(bs)
            dev_strs.append(f"{v:.4f}" if v is not None else "N/A")
        regime = m.get("regime", {})
        coup_regime = regime.get("coupling_regime", "N/A")
        dev_regime  = regime.get("device_regime", "N/A")
        lines.append(f"| {r['display_name']} | {coup_str} FLOP/B | " +
                     " | ".join(dev_strs) + f" | {coup_regime} | {dev_regime} |")
    lines.append("")

    # ---- FLOPs breakdown ----
    lines.append("## FLOPs Breakdown")
    lines.append("")
    for r in all_results:
        f = r.get("flops", {})
        lines.append(f"### {r['display_name']}")
        if f.get("error"):
            lines.append(f"Error: {f['error']}")
        else:
            lines.append(f"- **Method:** `{f.get('method', 'unknown')}`")
            lines.append(f"- **Total FLOPs (B=1):** {_fmt_flops(f.get('total_flops'))}")
            lines.append(f"- **FLOPs per entry:** {_fmt_flops(f.get('flops_per_entry'))}")
            if f.get("warning"):
                lines.append(f"- **Warning:** {f['warning']}")
            ob = f.get("op_breakdown", {})
            if ob:
                lines.append("- **Op breakdown (B=1):**")
                for op, cnt in sorted(ob.items(), key=lambda x: -x[1]):
                    lines.append(f"  - `{op}`: {_fmt_flops(cnt)}")
        lines.append("")

    # ---- Errors ----
    errors_present = [r for r in all_results if r.get("errors")]
    if errors_present:
        lines.append("## Errors and Warnings")
        lines.append("")
        for r in errors_present:
            lines.append(f"### {r['display_name']}")
            for e in r["errors"]:
                lines.append(f"- {e}")
            lines.append("")

    lines.append("---")
    lines.append("*Generated by `model_inspection/inspect_models.py`*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect model resource requirements from model_catalog.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device",      default="cpu",
                   help="Device to run on: 'cpu' or 'cuda' (default: cpu)")
    p.add_argument("--batch-sizes", default="1,8,32,128,512",
                   help="Comma-separated batch sizes for timing and memory (default: 1,8,32,128,512)")
    p.add_argument("--catalog",     default=None,
                   help="Path to model_catalog.json (default: same dir as this script)")
    p.add_argument("--warmup",      type=int, default=10,
                   help="Warmup iterations before timing (default: 10)")
    p.add_argument("--iters",       type=int, default=50,
                   help="Timed iterations per batch-size (default: 50)")
    p.add_argument("--timeout-s",   type=float, default=60.0,
                   help="Per-model per-batchsize timing timeout in seconds (default: 60)")
    p.add_argument("--models",      default=None,
                   help="Comma-separated model IDs to inspect (default: all in catalog)")
    p.add_argument("--skip-timing", action="store_true", default=False,
                   help="Skip timing and activation memory; only collect static facts")
    p.add_argument("--output-dir",  default=None,
                   help="Directory for results JSON/Markdown (default: results/ next to script)")
    p.add_argument("--no-giant",    action="store_true", default=False,
                   help="Skip benchmark_giant_mlp (avoids 1.56 GiB load time on CPU)")
    return p.parse_args()


def main():
    args = parse_args()

    script_dir  = Path(__file__).resolve().parent
    catalog_dir = script_dir
    catalog_path = Path(args.catalog) if args.catalog else script_dir / "model_catalog.json"

    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    entries = catalog["models"]
    if args.models:
        wanted = set(x.strip() for x in args.models.split(","))
        entries = [e for e in entries if e["id"] in wanted]
        missing = wanted - {e["id"] for e in entries}
        if missing:
            print(f"Warning: model IDs not found in catalog: {missing}", file=sys.stderr)

    if args.no_giant:
        entries = [e for e in entries if e["id"] != "giant_mlp"]

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        device_str = "cpu"
    device = torch.device(device_str)

    # Run metadata
    run_meta: dict[str, Any] = {
        "timestamp":      datetime.now().isoformat(),
        "device":         device_str,
        "torch_version":  torch.__version__,
        "python_version": platform.python_version(),
        "hostname":       platform.node(),
        "batch_sizes":    batch_sizes,
        "warmup":         args.warmup,
        "iters":          args.iters,
        "timeout_s":      args.timeout_s,
        "skip_timing":    args.skip_timing,
    }
    if device.type == "cuda":
        run_meta["cuda_device_name"]      = torch.cuda.get_device_name(device)
        run_meta["cuda_device_count"]     = torch.cuda.device_count()
        total_mem = torch.cuda.get_device_properties(device).total_memory
        run_meta["cuda_total_memory_bytes"] = total_mem
        run_meta["cuda_total_memory_gib"]   = round(total_mem / (1024**3), 2)
        run_meta["cuda_tf32_enabled"]       = torch.backends.cuda.matmul.allow_tf32
        run_meta["cuda_cudnn_tf32"]         = torch.backends.cudnn.allow_tf32

    print(f"Model Inspection — device={device_str}, batch_sizes={batch_sizes}")
    print(f"Catalog: {catalog_path}")
    print(f"Models:  {[e['id'] for e in entries]}")
    print()

    all_results = []
    for entry in entries:
        model_id = entry["id"]
        print(f"  [{model_id}] Inspecting ...", flush=True)
        try:
            r = inspect_one_model(
                entry, catalog_dir, device, batch_sizes,
                args.warmup, args.iters, args.timeout_s, args.skip_timing,
            )
        except Exception as exc:
            print(f"    ERROR: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            r = {
                "model_id": model_id, "display_name": entry.get("display_name", model_id),
                "family": entry.get("family"), "device": device_str,
                "errors": [f"Unhandled exception: {exc}"],
                "artifact": {}, "parameters": {}, "io": {}, "flops": {},
                "memory": {}, "timing": {}, "metrics": {},
            }
        all_results.append(r)
        p = r.get("parameters", {})
        print(f"    params={p.get('trainable_param_count','?'):,}  "
              f"payload={_fmt_bytes(p.get('total_payload_bytes'))}  "
              f"flops/entry={_fmt_flops(r.get('metrics',{}).get('flops_per_entry'))}")
        if r.get("errors"):
            for e in r["errors"]:
                print(f"    WARN: {e}", file=sys.stderr)

    # Output
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_{device_str}"

    json_path = output_dir / f"inspection_{tag}.json"
    md_path   = output_dir / f"inspection_{tag}.md"

    final = {"run_meta": run_meta, "results": all_results}
    with open(json_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nJSON  -> {json_path}")

    md = generate_markdown(all_results, run_meta, batch_sizes)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"MD    -> {md_path}")

    # Quick classification summary
    print(f"\n=== Classification Summary (ridge: FP32 non-TC ~{H100_NVL_RIDGE_FP32_NONTR:.0f} FLOP/B | TF32 TC ~{H100_NVL_RIDGE_TF32:.0f} FLOP/B) ===")
    for r in all_results:
        m = r.get("metrics", {})
        regime = m.get("regime", {})
        coup_oi = m.get("coupling_oi_flop_per_byte")
        coup_oi_str = f"{coup_oi:.2f}" if coup_oi is not None else "N/A"
        print(f"  {r['display_name'][:45]:45s}  coupling_OI={coup_oi_str:>10} FLOP/B  "
              f"=> {regime.get('coupling_regime', 'unknown')}")


if __name__ == "__main__":
    main()
