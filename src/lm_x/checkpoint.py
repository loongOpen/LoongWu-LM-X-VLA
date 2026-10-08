"""Local checkpoint manifest validation for inference startup."""

from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath


class CheckpointValidationError(ValueError):
    """Raised when a local checkpoint is incomplete or contains pointer stubs."""


_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")
_MODEL_ID_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_BACKBONE_LAYER = re.compile(r"^backbone\.model\.(?:model\.)?language_model\.layers\.(\d+)\.")


def _reject_lfs_pointer(path: Path) -> None:
    try:
        if path.stat().st_size == 0:
            raise CheckpointValidationError(f"Checkpoint file is empty: {path}")
        with path.open("rb") as stream:
            header = stream.read(len(_LFS_HEADER))
    except OSError as exc:
        raise CheckpointValidationError(f"Cannot read checkpoint file: {path}") from exc
    if header == _LFS_HEADER:
        raise CheckpointValidationError(
            f"Checkpoint file is a Git LFS pointer, not downloaded content: {path}"
        )


def read_json_object(path: Path) -> dict:
    """Read materialized JSON metadata before constructing any model."""
    _reject_lfs_pointer(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"Invalid JSON metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"JSON metadata must be an object: {path}")
    return payload


def _load_weight_index(path: Path) -> set[str]:
    payload = read_json_object(path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointValidationError(f"Weight index has an empty weight_map: {path}")
    if any(not isinstance(name, str) or not name.strip() for name in weight_map.values()):
        raise CheckpointValidationError(f"Weight index contains an invalid shard name: {path}")
    shards = set(weight_map.values())
    if any(
        Path(name).is_absolute()
        or PureWindowsPath(name).drive
        or "\\" in name
        or name in {".", ".."}
        or ".." in Path(name).parts
        for name in shards
    ):
        raise CheckpointValidationError(f"Weight index contains an unsafe shard path: {path}")
    return shards


def _looks_like_remote_model_id(value: str) -> bool:
    """Return whether ``value`` has the shape of a Hub model identifier."""

    if not value or value != value.strip() or value.endswith(".git"):
        return False
    if value.startswith((".", "~", "/", "\\")) or "\\" in value:
        return False
    if len(value) > 96 or "--" in value or ".." in value:
        return False
    components = value.split("/")
    return len(components) in {1, 2} and all(
        component and _MODEL_ID_COMPONENT.fullmatch(component) for component in components
    )


def validate_checkpoint_manifest(path: str | Path) -> Path | str:
    """Validate files required to load a checkpoint and its processor.

    Processor metadata may live beside ``config.json`` or in a ``processor``
    subdirectory. A local checkpoint returns its resolved root. A remote model
    identifier is returned unchanged because its files are validated by the
    loading backend after resolution.
    """

    local_candidate = Path(path).expanduser()
    if not local_candidate.exists():
        if isinstance(path, str) and _looks_like_remote_model_id(path):
            return path
        raise CheckpointValidationError(
            f"Checkpoint directory does not exist: {local_candidate.resolve()}"
        )
    if not local_candidate.is_dir():
        raise CheckpointValidationError(f"Checkpoint path is not a directory: {local_candidate}")
    root = local_candidate.resolve()

    config_file = root / "config.json"
    if not config_file.is_file():
        raise CheckpointValidationError(f"Missing model config: {config_file}")
    read_json_object(config_file)

    checked_weights: list[Path] = []
    for name in _WEIGHT_FILES:
        candidate = root / name
        if candidate.is_file():
            _reject_lfs_pointer(candidate)
            checked_weights.append(candidate)

    for name in _WEIGHT_INDEX_FILES:
        index_file = root / name
        if not index_file.is_file():
            continue
        checked_weights.append(index_file)
        for shard_name in _load_weight_index(index_file):
            shard = root / shard_name
            if not shard.is_file():
                raise CheckpointValidationError(
                    f"Weight index {index_file.name} references missing shard: {shard_name}"
                )
            _reject_lfs_pointer(shard)
            checked_weights.append(shard)

    if not checked_weights:
        expected = ", ".join((*_WEIGHT_FILES, *_WEIGHT_INDEX_FILES))
        raise CheckpointValidationError(
            f"Missing model weights in {root}; expected one of: {expected}"
        )
    for weight_file in checked_weights:
        _reject_lfs_pointer(weight_file)

    processor_root = root
    if not (root / "processor_config.json").is_file() and (root / "processor").is_dir():
        processor_root = root / "processor"
    missing = [name for name in _PROCESSOR_FILES if not (processor_root / name).is_file()]
    if missing:
        raise CheckpointValidationError(
            f"Missing processor metadata in {processor_root}: {', '.join(missing)}"
        )
    for name in _PROCESSOR_FILES:
        read_json_object(processor_root / name)

    return root


def validate_model_loading_info(
    loading_info: dict,
    *,
    model_reference: str,
    select_layer: int,
) -> None:
    """Reject checkpoint/model mismatches that would corrupt action inference.

    Missing backbone tensors are allowed because the backbone is loaded independently before
    checkpoint overlays are applied. Backbone language layers at or above ``select_layer`` may
    also appear as unexpected when a checkpoint stores more layers than the runtime retains.
    Every action head tensor and every configured optional branch must load exactly.
    """

    missing_keys = [str(key) for key in loading_info.get("missing_keys", ())]
    unexpected_keys = [str(key) for key in loading_info.get("unexpected_keys", ())]
    mismatched_keys = [str(key) for key in loading_info.get("mismatched_keys", ())]
    backend_errors = [str(message) for message in loading_info.get("error_msgs", ())]

    disallowed_missing = [key for key in missing_keys if not key.startswith("backbone.")]
    disallowed_unexpected: list[str] = []
    for key in unexpected_keys:
        layer_match = _BACKBONE_LAYER.match(key)
        if layer_match is not None and int(layer_match.group(1)) >= select_layer:
            continue
        disallowed_unexpected.append(key)

    errors: list[str] = []
    if disallowed_missing:
        errors.append(f"missing keys ({len(disallowed_missing)}): {disallowed_missing}")
    if disallowed_unexpected:
        errors.append(f"unexpected keys ({len(disallowed_unexpected)}): {disallowed_unexpected}")
    if mismatched_keys:
        errors.append(f"mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
    if backend_errors:
        errors.append(f"loader errors ({len(backend_errors)}): {backend_errors}")
    if errors:
        raise CheckpointValidationError(
            "Checkpoint weights are incompatible with the inference graph for "
            f"{model_reference!r}:\n" + "\n".join(errors)
        )


__all__ = [
    "CheckpointValidationError",
    "validate_checkpoint_manifest",
    "validate_model_loading_info",
]
