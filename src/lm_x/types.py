"""Stable data contracts used by the LM-X inference boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import numpy as np

from .embodiment_tags import EmbodimentTag


class MessageType(Enum):
    EPISODE_STEP = "episode_step"


class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"


class ActionType(Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(Enum):
    DEFAULT = "default"
    XYZ_ROT6D = "xyz+rot6d"
    XYZ_ROTVEC = "xyz+rotvec"


EnumT = TypeVar("EnumT", bound=Enum)


def _enum_value(enum_type: type[EnumT], raw: EnumT | str, field_name: str) -> EnumT:
    """Accept the member names stored in checkpoint JSON and normal enum values."""
    if isinstance(raw, enum_type):
        return raw
    if not isinstance(raw, str):
        raise TypeError(
            f"{field_name} must be {enum_type.__name__} or str, got {type(raw).__name__}"
        )
    try:
        return enum_type[raw]
    except KeyError:
        try:
            return enum_type(raw)
        except ValueError as exc:
            choices = ", ".join(member.name for member in enum_type)
            raise ValueError(f"Unknown {field_name} {raw!r}; expected one of: {choices}") from exc


def _list_field(name: str, value: Any, *, allow_empty: bool) -> list:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{name} must be {qualifier}, got {value!r}")
    return value


@dataclass
class LMXStepData:
    """One pre-collation observation consumed by :class:`LMXProcessor`."""

    images: dict[str, list[np.ndarray]]
    states: dict[str, np.ndarray]
    masks: dict[str, list[np.ndarray]] | None = None
    text: str | None = None
    embodiment: EmbodimentTag = field(default_factory=lambda: next(iter(EmbodimentTag)))


@dataclass
class ActionConfig:
    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat
    state_key: str | None = None

    def __post_init__(self) -> None:
        self.rep = _enum_value(ActionRepresentation, self.rep, "action representation")
        self.type = _enum_value(ActionType, self.type, "action type")
        self.format = _enum_value(ActionFormat, self.format, "action format")
        if self.state_key is not None and not isinstance(self.state_key, str):
            raise TypeError("state_key must be a string or None")

    @classmethod
    def from_raw(cls, value: ActionConfig | dict[str, Any]) -> ActionConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                f"action config must be a mapping or ActionConfig, got {type(value).__name__}"
            )
        required = ("rep", "type", "format")
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"action config is missing fields: {', '.join(missing)}")
        return cls(
            rep=value["rep"],
            type=value["type"],
            format=value["format"],
            state_key=value.get("state_key"),
        )


@dataclass
class ModalityConfig:
    """Modality schema serialized in ``processor_config.json``."""

    delta_indices: list[int]
    modality_keys: list[str]
    sin_cos_embedding_keys: list[str] | None = None
    mean_std_embedding_keys: list[str] | None = None
    action_configs: list[ActionConfig] | None = None

    def __post_init__(self) -> None:
        self.delta_indices = _list_field("delta_indices", self.delta_indices, allow_empty=True)
        self.modality_keys = _list_field("modality_keys", self.modality_keys, allow_empty=False)

        for name in ("sin_cos_embedding_keys", "mean_std_embedding_keys"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _list_field(name, value, allow_empty=True))

        if self.action_configs is None:
            return
        configs = _list_field("action_configs", self.action_configs, allow_empty=True)
        if len(configs) != len(self.modality_keys):
            raise ValueError(
                "Number of action configs "
                f"({len(configs)}) must match number of modality keys ({len(self.modality_keys)})"
            )
        self.action_configs = [ActionConfig.from_raw(config) for config in configs]
