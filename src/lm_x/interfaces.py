"""Processor contracts used by the LM-X inference boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from transformers import ProcessorMixin

from lm_x.types import EmbodimentTag, ModalityConfig


class InferenceProcessor(ProcessorMixin, ABC):
    """Minimal surface a checkpoint-backed observation/action processor must expose."""

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return self.encode_messages(messages)

    def encode_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        return self.action_to_robot(action, embodiment_tag, state)

    def action_to_robot(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @property
    def collator(self):
        raise NotImplementedError

    @abstractmethod
    def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
        raise NotImplementedError

    def eval(self):
        return self

    def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        configs = getattr(self, "modality_configs", None)
        if configs is None:
            raise AttributeError(f"{type(self).__name__} has no modality_configs")
        return configs
