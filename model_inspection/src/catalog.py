"""Model catalog loading and artifact/input-contract helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


def load_catalog(path: Path) -> dict[str, Any]:
    import json

    catalog = json.loads(path.read_text())
    ids = [entry["id"] for entry in catalog["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate model IDs in {path}")
    return catalog


def select_models(catalog: dict[str, Any], requested: str) -> list[str]:
    if requested in (None, "all"):
        return [entry["id"] for entry in catalog["models"]]
    models = [name.strip() for name in requested.split(",") if name.strip()]
    known = {entry["id"] for entry in catalog["models"]}
    unknown = set(models) - known
    if unknown:
        raise ValueError(f"unknown model IDs: {sorted(unknown)}")
    return models


def spec_for(catalog: dict[str, Any], model_id: str) -> dict[str, Any]:
    return next(entry for entry in catalog["models"] if entry["id"] == model_id)


def replica_root() -> Path:
    import os

    default = f"/tmp/{os.environ.get('USER', 'user')}/model_inspection_replicas"
    return Path(os.environ.get("MODEL_REPLICA_ROOT", default)).expanduser()


def _artifact_path(catalog_dir: Path, value: str) -> Path:
    expanded = value.replace("{replica_root}", str(replica_root()))
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (catalog_dir / path).resolve()


def resolve_artifact(catalog_dir: Path, spec: dict[str, Any], device: torch.device) -> Path:
    preferred = "artifact_cuda" if device.type == "cuda" else "artifact_cpu"
    attempted: list[Path] = []
    build_errors: list[str] = []
    for key in dict.fromkeys((preferred, "artifact_cpu", "artifact_cuda")):
        value = spec.get(key)
        if not value:
            continue
        path = _artifact_path(catalog_dir, value)
        if path not in attempted:
            attempted.append(path)
        if path.exists():
            return path
        factory = spec.get("replica")
        if factory and not (device.type == "cuda" and not torch.cuda.is_available()):
            try:
                import importlib

                builder = importlib.import_module("build_model_replicas")
                builder.ensure_artifact(factory, device.type, path)
            except Exception as exc:
                build_errors.append(f"    - {path}: {exc}")
            if path.exists():
                return path
    tried = "\n".join(f"    - {path}" for path in attempted) or "    (no artifact path configured)"
    detail = ""
    if build_errors:
        detail = "\n  replica build failures:\n" + "\n".join(build_errors)
    raise FileNotFoundError(
        f"no artifact available for model '{spec.get('id', '?')}' on {device.type}; tried:\n{tried}{detail}"
    )


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in flatten_tensors(child)]
    raise TypeError(f"unsupported model output type: {type(value).__name__}")


def tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def make_inputs(spec: dict[str, Any], batch_size: int, device: torch.device, pinned: bool = False, fill: float = 0.1) -> tuple[torch.Tensor, ...]:
    tensors = []
    for field in spec["inputs"]:
        shape = [batch_size if dim is None else dim for dim in field["shape"]]
        tensors.append(torch.full(shape, fill, dtype=DTYPES[field["dtype"]], device=device, pin_memory=pinned))
    return tuple(tensors)


def schema_bytes(fields: list[dict[str, Any]]) -> int:
    total = 0
    for field in fields:
        elements = 1
        for dim in field["shape"][1:]:
            elements *= 1 if dim is None else dim
        total += elements * torch.empty((), dtype=DTYPES[field["dtype"]]).element_size()
    return total


def field_numel(field: dict[str, Any]) -> int:
    elements = 1
    for dim in field["shape"][1:]:
        elements *= 1 if dim is None else dim
    return elements
