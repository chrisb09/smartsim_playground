#!/usr/bin/env python3
"""
model_inspection/build_test_mlp_artifacts.py
=============================================
Generate TorchScript artifacts for the test-only MMCP-interface MLP.

Two variants are exported:
  - mmcp_test_mlp_m5:  [B, 5,  512] -> [B, 2, 512]
  - mmcp_test_mlp_m10: [B, 10, 512] -> [B, 2, 512]

Both match the MMCP transformer's output contract ([B, 2, 512]) while being
intentionally much smaller and faster, for use as compute/bandwidth profiling
baselines without the cost of loading the real transformer artifact.

Architecture (shared temporal encoder):
  - Step projection: Linear(512, hidden) -> GELU, applied independently to each step
  - Flatten:         [B, m, hidden] -> [B, m * hidden]
  - Decoder:         Linear(m * hidden, 2 * 512) -> reshape to [B, 2, 512]

At hidden=64:
  - m=5:  params = 512*64 + 64 + 5*64*1024 + 1024 = 33,856 + 329,728 = 363,584 (wait: recount)
  - Actual: step_proj: 512*64 + 64 = 32,832; decoder: (5*64)*1024 + 1024 = 328,704; total = 361,536
  - m=10: step_proj: 32,832; decoder: (10*64)*1024 + 1024 = 655,360 + 1024 = 656,384; total = 689,216

FLOPs per entry (B=1, using FlopCounterMode):
  - step_proj (per step): 2 * 512 * 64 = 65,536; times m steps
  - GELU: not counted by FlopCounterMode
  - decoder: 2 * (m*64) * 1024
  - m=5:  5 * 65,536 + 2 * 320 * 1024 = 327,680 + 655,360 = 983,040
  - m=10: 10 * 65,536 + 2 * 640 * 1024 = 655,360 + 1,310,720 = 1,966,080

Usage:
  python build_test_mlp_artifacts.py [--device cpu|cuda|both] [--hidden 64]
                                     [--output-dir artifacts]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMCPTestMLP(nn.Module):
    """
    Test-only MLP matching the MMCP transformer I/O contract.

    Input:  [B, m, 512]   (m history steps, each a flattened 8^3 = 512 feature cube)
    Output: [B, 2, 512]   (2 forecast steps matching mmcp_transformer output)

    Architecture: shared per-step encoder (Linear -> GELU), flatten, single decoder.
    No training path; no dropout; no positional encoding.
    """

    def __init__(self, m: int = 5, hidden: int = 64):
        super().__init__()
        if m <= 0:
            raise ValueError("m must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.m = m
        self.hidden = hidden
        # Shared linear applied to each of the m steps
        self.step_proj = nn.Linear(512, hidden)
        # Decoder: flatten m encoded steps -> 2 output steps of 512 features
        self.decoder = nn.Linear(m * hidden, 2 * 512)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [B, m, 512]
        B, m, _ = src.shape
        # Apply shared step projection across the sequence dimension
        h = F.gelu(self.step_proj(src))   # [B, m, hidden]
        h = h.reshape(B, m * self.hidden)  # [B, m * hidden]
        out = self.decoder(h)              # [B, 2 * 512]
        return out.reshape(B, 2, 512)      # [B, 2, 512]


def export_artifact(model: MMCPTestMLP, device: torch.device, output_path: str) -> dict:
    """Script and save the model to the given path. Returns artifact metadata."""
    model = model.to(device).eval()
    example = torch.zeros(1, model.m, 512, device=device)

    # Try script first, fall back to trace
    try:
        scripted = torch.jit.script(model)
    except Exception as script_exc:
        print(f"  torch.jit.script failed ({script_exc}), falling back to trace ...", file=sys.stderr)
        scripted = torch.jit.trace(model, (example,), check_trace=True)

    # Verify output shape before saving
    with torch.inference_mode():
        out = scripted(example)
    assert out.shape == torch.Size([1, 2, 512]), f"Unexpected output shape: {out.shape}"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scripted.save(output_path)

    file_bytes = os.path.getsize(output_path)
    param_count = sum(p.numel() for p in model.parameters())
    payload_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    print(f"  Saved {output_path}  ({file_bytes/1024:.1f} KiB, {param_count:,} params)")
    return {
        "path": os.path.abspath(output_path),
        "size_bytes": file_bytes,
        "param_count": param_count,
        "payload_bytes": payload_bytes,
        "device": str(device),
        "m": model.m,
        "hidden": model.hidden,
    }


def main():
    p = argparse.ArgumentParser(description="Build test-only MMCP-interface MLP artifacts.")
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda", "both"],
                   help="Target device(s) for artifact export (default: cpu)")
    p.add_argument("--hidden",     type=int, default=64,
                   help="Encoder hidden width per step (default: 64)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for .pt artifacts (default: artifacts/ next to this script)")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir    = Path(args.output_dir) if args.output_dir else script_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    devices_to_export: list[torch.device] = []
    if args.device in ("cpu", "both"):
        devices_to_export.append(torch.device("cpu"))
    if args.device in ("cuda", "both"):
        if torch.cuda.is_available():
            devices_to_export.append(torch.device("cuda"))
        else:
            print("Warning: CUDA requested but not available; skipping CUDA export.", file=sys.stderr)

    manifest: list[dict] = []
    for m_val in (5, 10):
        model = MMCPTestMLP(m=m_val, hidden=args.hidden)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nMMCPTestMLP m={m_val}, hidden={args.hidden}: {total_params:,} parameters")

        for device in devices_to_export:
            dev_tag = "cuda" if device.type == "cuda" else "cpu"
            fname   = f"mmcp_test_mlp_m{m_val}_{dev_tag}.pt"
            fpath   = str(out_dir / fname)
            meta    = export_artifact(model, device, fpath)
            meta["model_id"] = f"mmcp_test_mlp_m{m_val}"
            manifest.append(meta)

    manifest_path = out_dir / "mmcp_test_mlp_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"version": 1, "artifacts": manifest}, f, indent=2, default=str)
    print(f"\nManifest -> {manifest_path}")


if __name__ == "__main__":
    main()
