"""Checkpoint-driven state normalization and action reconstruction."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import numpy as np

from lm_x.data_utils import (
    decode_modality_configs,
    destandardize,
    encode_periodic,
    scale_from_unit,
    scale_to_unit,
    standardize,
    tree_arrays,
)
from lm_x.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from lm_x.state_action.pose import EndEffectorPose, JointPose
from lm_x.types import ActionFormat, ActionRepresentation, ActionType, ModalityConfig

logger = logging.getLogger(__name__)

Statistics = dict[str, dict[str, dict[str, dict[str, list[float]]]]]
NormParams = dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]


class _NormalizationCatalog:
    """Compile raw checkpoint statistics into array parameters used per request."""

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        *,
        use_percentiles: bool,
        use_relative_action: bool,
    ):
        self.configs = modality_configs
        self.lower_key = "q01" if use_percentiles else "min"
        self.upper_key = "q99" if use_percentiles else "max"
        self.use_relative_action = use_relative_action

    def compile(self, statistics: Statistics) -> NormParams:
        compiled: NormParams = {}
        for embodiment, by_modality in statistics.items():
            compiled[embodiment] = {}
            for modality in ("state", "action"):
                groups = by_modality.get(modality)
                if groups is None:
                    continue
                compiled[embodiment][modality] = {
                    name: self._group_params(raw) for name, raw in groups.items()
                }
            self._apply_relative_overrides(embodiment, by_modality, compiled[embodiment])
        return compiled

    def _group_params(self, raw: dict[str, list[float]]) -> dict[str, np.ndarray]:
        lower = np.asarray(raw[self.lower_key])
        upper = np.asarray(raw[self.upper_key])
        return {
            "min": lower,
            "max": upper,
            "dim": np.asarray(lower.shape[-1]),
            "mean": np.asarray(raw["mean"]),
            "std": np.asarray(raw["std"]),
        }

    def _apply_relative_overrides(
        self,
        embodiment: str,
        raw: dict[str, dict[str, dict[str, list[float]]]],
        compiled: dict[str, dict[str, dict[str, np.ndarray]]],
    ) -> None:
        if not self.use_relative_action:
            return
        action_schema = self.configs.get(embodiment, {}).get("action")
        if action_schema is None or action_schema.action_configs is None:
            return
        relative_groups = raw.get("relative_action")
        for key, config in zip(action_schema.modality_keys, action_schema.action_configs):
            if config.rep is not ActionRepresentation.RELATIVE:
                continue
            if relative_groups is None or key not in relative_groups:
                raise ValueError(f"Relative action statistics required for {embodiment!r}/{key!r}")
            dimension = compiled["action"][key]["dim"]
            replacement = tree_arrays(relative_groups[key])
            replacement["dim"] = dimension
            compiled["action"][key] = replacement


class _ActionReconstructor:
    @staticmethod
    def convert(
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        trajectory = np.asarray(action)
        reference = np.asarray(reference_state)
        if trajectory.ndim != 2:
            raise ValueError(f"Expected action shape (T, D), got {trajectory.shape}")
        if reference.ndim != 1:
            raise ValueError(f"Expected state shape (D,), got {reference.shape}")
        if reference.shape[0] != trajectory.shape[1]:
            raise ValueError(f"State dim {reference.shape[0]} != action dim {trajectory.shape[1]}")

        if action_type is ActionType.EEF:
            relative = EndEffectorActionChunk.from_array(trajectory, action_format)
            anchor = EndEffectorPose.from_action_format(reference, action_format)
        elif action_type is ActionType.NON_EEF:
            relative = JointActionChunk([JointPose(row) for row in trajectory])
            anchor = JointPose(reference)
        else:
            raise ValueError(f"Unknown ActionType: {action_type}")
        return relative.rebase(anchor).encode(action_format)


class StateActionProcessor:
    """Inference facade for state encoding and action decoding."""

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: Statistics | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        self.modality_configs = decode_modality_configs(modality_configs)
        self.statistics: Statistics = {}
        self.use_percentiles = bool(use_percentiles)
        self.clip_outliers = bool(clip_outliers)
        self.apply_sincos_state_encoding = bool(apply_sincos_state_encoding)
        self.use_relative_action = bool(use_relative_action)
        self.norm_params: NormParams = {}
        self._catalog = _NormalizationCatalog(
            self.modality_configs,
            use_percentiles=self.use_percentiles,
            use_relative_action=self.use_relative_action,
        )
        if statistics is not None:
            self.set_statistics(statistics)

    def eval(self) -> StateActionProcessor:
        return self

    def set_statistics(self, statistics: Statistics, override: bool = False) -> None:
        for embodiment, values in statistics.items():
            if embodiment in self.statistics and not override:
                logger.warning(
                    "Statistics for embodiment %r already exist; keeping checkpoint values",
                    embodiment,
                )
                continue
            self.statistics[embodiment] = deepcopy(values)
        self.norm_params = self._catalog.compile(self.statistics)

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        schema = self.modality_configs[embodiment_tag]["state"]
        periodic_keys = (
            set(schema.sin_cos_embedding_keys or []) if self.apply_sincos_state_encoding else set()
        )
        standard_keys = set(schema.mean_std_embedding_keys or [])
        encoded: dict[str, np.ndarray] = {}
        for key in schema.modality_keys:
            if key not in state:
                raise KeyError(
                    f"Joint group {key!r} not found in state for embodiment {embodiment_tag!r}"
                )
            values = state[key]
            if key in periodic_keys:
                encoded[key] = encode_periodic(values)
                continue
            params = self.norm_params[embodiment_tag]["state"][key]
            normalized = (
                standardize(values, params)
                if key in standard_keys
                else scale_to_unit(values, params)
            )
            if self.clip_outliers and key not in standard_keys:
                normalized = np.clip(normalized, -1.0, 1.0)
            encoded[key] = normalized
        return encoded

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        schema = self.modality_configs[embodiment_tag]["action"]
        standard_keys = set(schema.mean_std_embedding_keys or [])
        decoded: dict[str, np.ndarray] = {}
        for key in schema.modality_keys:
            if key not in action:
                raise KeyError(
                    f"Joint group {key!r} not found in action for embodiment {embodiment_tag!r}"
                )
            params = self.norm_params[embodiment_tag]["action"][key]
            decoded[key] = (
                destandardize(action[key], params)
                if key in standard_keys
                else scale_from_unit(action[key], params)
            )

        if not self.use_relative_action or schema.action_configs is None:
            return decoded
        for key, config in zip(schema.modality_keys, schema.action_configs):
            if config.rep is not ActionRepresentation.RELATIVE:
                continue
            if state is None:
                raise ValueError(
                    f"State is required to reconstruct relative action {embodiment_tag!r}/{key!r}"
                )
            state_key = config.state_key or key
            if state_key not in state:
                raise KeyError(
                    f"Reference state {state_key!r} not found for embodiment {embodiment_tag!r}"
                )
            decoded[key] = self._reconstruct_batch(
                decoded[key],
                state[state_key],
                config.type,
                config.format,
            )
        return decoded

    @staticmethod
    def _reconstruct_batch(
        action: np.ndarray,
        state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        action_array = np.asarray(action)
        state_array = np.asarray(state)
        was_batched = action_array.ndim == 3
        if action_array.ndim not in (2, 3):
            raise ValueError(f"relative action must be rank 2 or 3, got {action_array.shape}")
        if not was_batched:
            action_array = action_array[None, ...]
        if state_array.ndim == 2:
            state_array = state_array[None, ...]
        if state_array.ndim != 3 or state_array.shape[0] != action_array.shape[0]:
            raise ValueError(
                "State/action batch mismatch while reconstructing relative action: "
                f"state={state_array.shape}, action={action_array.shape}"
            )
        result = np.stack(
            [
                _ActionReconstructor.convert(values, history[-1], action_type, action_format)
                for history, values in zip(state_array, action_array)
            ]
        )
        return result if was_batched else result[0]

    def get_action_dim(self, embodiment_tag: str) -> int:
        groups = self.modality_configs[embodiment_tag]["action"].modality_keys
        return sum(
            int(np.asarray(self.norm_params[embodiment_tag]["action"][key]["dim"]).item())
            for key in groups
        )

    def relative_to_absolute(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        return _ActionReconstructor.convert(
            action,
            reference_state,
            action_type,
            action_format,
        )

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Compatibility wrapper for runtime 0.2 integrations."""
        return self.relative_to_absolute(action, reference_state, action_type, action_format)

    def __str__(self) -> str:
        settings: dict[str, Any] = {
            "use_percentiles": self.use_percentiles,
            "clip_outliers": self.clip_outliers,
            "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
            "use_relative_action": self.use_relative_action,
        }
        return (
            f"StateActionProcessor(embodiments={sorted(self.modality_configs)}, "
            f"settings={settings})"
        )
