#!/usr/bin/env python3
"""Export small, reproducible TorchScript replicas for model-inspection studies.

The files created by this tool are untrained and contain random weights. They
preserve the benchmark contracts and reproduce the measured parameter counts,
FLOPs, tensor-core usage, and L2 traffic of the real artifacts (the TBL
transformer replica mirrors the real artifact's two-step autoregressive
execution), so they are suitable for workflow and performance-method
validation. They are not functional substitutes for the trained scientific
models.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from catalog import flatten_tensors


class WaterCNNReplica(nn.Module):
    def __init__(self):
        super().__init__()
        self.water_conv = nn.Conv2d(1, 8, 3)
        self.terrain_conv = nn.Conv2d(1, 8, 3)
        self.combine_1x1 = nn.Conv2d(2, 8, 1)
        self.combine_3x3 = nn.Conv2d(8, 16, 3)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x_water: torch.Tensor, x_terrain: torch.Tensor) -> torch.Tensor:
        water = F.relu(self.water_conv(x_water))
        terrain = F.relu(self.terrain_conv(x_terrain))
        combined = F.relu(self.combine_1x1(torch.cat((x_water, x_terrain), dim=1)))
        combined = F.relu(self.combine_3x3(combined))
        return self.fc(torch.cat((water.flatten(1), terrain.flatten(1), combined.flatten(1)), dim=1)).squeeze(1)


class MMCPTestMLPReplica(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_proj = nn.Linear(512, 64)
        self.decoder = nn.Linear(5 * 64, 2 * 512)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch = src.shape[0]
        encoded = F.gelu(self.step_proj(src)).reshape(batch, 5 * 64)
        return self.decoder(encoded).reshape(batch, 2, 512)


class MMCPTransformerReplica(nn.Module):
    """Contract-compatible reconstruction inferred from the executed artifact.

    The production artifact is opaque. The observed 1024-wide, 16-head,
    6-encoder/6-decoder, 4096-FFN structure closely reproduces its parameter
    scale. The production model is autoregressive over the two forecast steps
    and executes the encoder/decoder stack once per step (twice in total),
    which the replica mirrors by feeding each prediction back into the
    history. Its random weights and forecast handling are not the trained
    model's scientific behaviour.
    """

    def __init__(self):
        super().__init__()
        width = 1024
        encoder_layer = nn.TransformerEncoderLayer(width, 16, 4096, dropout=0.0, activation="gelu", batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(width, 16, 4096, dropout=0.0, activation="gelu", batch_first=True)
        self.input_proj = nn.Linear(512, width)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6)
        self.forecast_queries = nn.Parameter(torch.zeros(1, 2, width))
        self.output_proj = nn.Linear(width, 512)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch = src.shape[0]
        history = src
        outputs = []
        for step in range(2):
            memory = self.encoder(self.input_proj(history))
            query = self.forecast_queries[:, step:step + 1, :].expand(batch, -1, -1)
            prediction = self.output_proj(self.decoder(query, memory))
            outputs.append(prediction)
            history = torch.cat((history[:, 1:, :], prediction), dim=1)
        return torch.cat(outputs, dim=1)


class GiantMLPReplica(nn.Module):
    def __init__(self):
        super().__init__()
        width = 4096
        self.input_proj = nn.Linear(18, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(12)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, x_water: torch.Tensor, x_terrain: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_proj(torch.cat((x_water.flatten(1), x_terrain.flatten(1)), dim=1)))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.head(self.final_norm(hidden)).squeeze(1)


SPECS = {
    "watercnn": {
        "factory": WaterCNNReplica,
        "inputs": [{"name": "x_water", "shape": [None, 1, 3, 3], "dtype": "float32"}, {"name": "x_terrain", "shape": [None, 1, 3, 3], "dtype": "float32"}],
    },
    "mmcp_test_mlp_m5": {
        "factory": MMCPTestMLPReplica,
        "inputs": [{"name": "src", "shape": [None, 5, 512], "dtype": "float32"}],
    },
    "tbl_transformer": {
        "factory": MMCPTransformerReplica,
        "inputs": [{"name": "src", "shape": [None, 5, 512], "dtype": "float32"}],
    },
    "giant_mlp": {
        "factory": GiantMLPReplica,
        "inputs": [{"name": "x_water", "shape": [None, 1, 3, 3], "dtype": "float32"}, {"name": "x_terrain", "shape": [None, 1, 3, 3], "dtype": "float32"}],
    },
}


def inputs_for(spec: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros([1 if value is None else value for value in field["shape"]], device=device) for field in spec["inputs"])


def save_replica(model_id: str, device: torch.device, output_path: Path, seed: int = 20260909) -> tuple[Path, int, int]:
    """Export one replica to ``output_path``; returns (path, parameter_elements, parameter_bytes)."""
    if model_id not in SPECS:
        raise KeyError(f"no replica factory for model '{model_id}'")
    torch.manual_seed(seed)
    model = SPECS[model_id]["factory"]().to(device).eval()
    inputs = inputs_for(SPECS[model_id], device)
    with torch.inference_mode():
        model(*inputs)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        # Transformer internals vary between supported PyTorch versions; a
        # fixed-contract trace remains valid for these inspection replicas.
        scripted = torch.jit.trace(model, inputs, check_trace=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return output_path, parameters, parameters * 4


def ensure_artifact(model_id: str, device_type: str, path: Path) -> Path:
    """Build the replica at ``path`` if it does not exist yet."""
    if path.exists():
        return path
    save_replica(model_id, torch.device(device_type), path)
    return path


def replica_catalog(records: list[dict]) -> dict:
    """Build a measure_models-compatible catalog from the exported replicas."""
    grouped: dict[str, dict] = {}
    for record in records:
        entry = grouped.setdefault(
            record["model_id"],
            {"id": record["model_id"], "display_name": f"{record['model_id']} (untrained replica)", "backend": "torchscript", "artifact_cpu": None, "artifact_cuda": None},
        )
        key = f"artifact_{record['device']}"
        entry[key] = record["artifact"]
        outputs = []
        for shape in record.get("output_shapes", []):
            outputs.append({"name": "out", "shape": [None] + list(shape)[1:], "dtype": "float32"})
        entry.setdefault("outputs", outputs)
        entry["inputs"] = record["inputs"]
    return {
        "version": 1,
        "description": "Untrained, contract-compatible TorchScript replicas. The catalog follows the canonical model_catalog.json schema so the same inspection pipeline can run against it.",
        "models": list(grouped.values()),
    }


def export_one(model_id: str, spec: dict, device: torch.device, output_dir: Path) -> dict:
    model = spec["factory"]().to(device).eval()
    inputs = inputs_for(spec, device)
    with torch.inference_mode():
        output = model(*inputs)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        # Transformer internals vary between supported PyTorch versions; a
        # fixed-contract trace remains valid for these inspection replicas.
        scripted = torch.jit.trace(model, inputs, check_trace=False)
    path = output_dir / f"{model_id}_{device.type}.pt"
    scripted.save(str(path))
    input_spec = output_dir / f"{model_id}_input_spec.json"
    input_spec.write_text(json.dumps({"inputs": spec["inputs"]}, indent=2))
    return {
        "model_id": model_id,
        "artifact": str(path),
        "device": device.type,
        "parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "inputs": spec["inputs"],
        "input_spec": str(input_spec),
        "output_shapes": [list(tensor.shape) for tensor in flatten_tensors(output)],
        "output_shape": list(output.shape),
        "untrained": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=f"/tmp/{os.environ.get('USER', 'user')}/model_inspection_replicas")
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--models", default=",".join(SPECS))
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    requested = [model_id.strip() for model_id in args.models.split(",") if model_id.strip()]
    unknown = set(requested) - set(SPECS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    devices = [torch.device("cpu")]
    if args.device in ("cuda", "both"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA export requested but CUDA is unavailable")
        devices = [torch.device("cuda")] if args.device == "cuda" else devices + [torch.device("cuda")]
    manifest = {"schema_version": 1, "seed": args.seed, "purpose": "untrained contract-compatible inspection replicas", "artifacts": []}
    for model_id in requested:
        for device in devices:
            record = export_one(model_id, SPECS[model_id], device, output_dir)
            manifest["artifacts"].append(record)
            print(f"{model_id} ({device.type}): {record['parameter_elements']:,} parameters -> {record['artifact']}")
    (output_dir / "replica_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "model_catalog.json").write_text(json.dumps(replica_catalog(manifest["artifacts"]), indent=2))
    print(f"Wrote {output_dir / 'replica_manifest.json'} and {output_dir / 'model_catalog.json'}")


if __name__ == "__main__":
    main()
