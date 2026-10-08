"""Shared fixtures that do not import the runtime or its heavy dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "lm_x"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON inside a test-owned temporary directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_lfs_pointer(*, oid: str = "0" * 64, size: int = 4096) -> bytes:
    """Return a syntactically valid Git LFS pointer, not materialized content."""

    return LFS_HEADER + b"\n" + f"oid sha256:{oid}\nsize {size}\n".encode("ascii")


def make_checkpoint(
    root: Path,
    *,
    processor_in_subdir: bool = False,
    include_weights: bool = True,
    sharded: bool = False,
    lfs_weight: bool = False,
) -> Path:
    """Create the smallest manifest-valid checkpoint tree for contract tests."""

    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "config.json",
        {
            "architectures": ["LMXModel"],
            "model_type": "longwu_lm_x",
        },
    )

    if include_weights and sharded:
        shard_names = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        write_json(
            root / "model.safetensors.index.json",
            {
                "metadata": {"total_size": 8192},
                "weight_map": {
                    "action_head.weight": shard_names[0],
                    "backbone.weight": shard_names[1],
                },
            },
        )
        for index, name in enumerate(shard_names, start=1):
            (root / name).write_bytes((f"materialized-shard-{index}".encode("ascii")) * 256)
    elif include_weights:
        payload = git_lfs_pointer() if lfs_weight else b"materialized-safetensors" * 256
        (root / "model.safetensors").write_bytes(payload)

    processor_root = root / "processor" if processor_in_subdir else root
    write_json(
        processor_root / "processor_config.json",
        {
            "processor_class": "LMXProcessor",
            "modality_configs": {},
        },
    )
    write_json(processor_root / "statistics.json", {})
    write_json(processor_root / "embodiment_id.json", {"new_embodiment": 10})
    return root
