#!/usr/bin/env python3
"""Minimal NCU/plain forward target: load one catalog model and run one marked forward.

Used internally by the NCU and Nsight Systems stages; not a public entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import load_catalog, resolve_artifact, spec_for
from inspection import configure_precision, run_ncu_forward, run_nsys_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "tf32"), default="fp32")
    parser.add_argument("--mode", choices=("ncu", "nsys"), default="ncu")
    parser.add_argument("--nsys-repeats", type=int, default=12)
    args = parser.parse_args()

    catalog_path = Path(args.catalog).resolve()
    catalog = load_catalog(catalog_path)
    spec = spec_for(catalog, args.model)
    device = torch.device(args.device)
    configure_precision(args.precision)
    artifact = resolve_artifact(catalog_path.parent, spec, device)
    model = torch.jit.load(str(artifact), map_location=device).eval()
    if args.mode == "ncu":
        run_ncu_forward(model, spec, args.batch, device)
    else:
        run_nsys_step(model, spec, args.batch, device, args.nsys_repeats)


if __name__ == "__main__":
    main()
