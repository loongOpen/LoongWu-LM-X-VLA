"""Minimal policy lifecycle shared by local and remote inference implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _PolicyInvocation:
    """Execute one policy request with optional boundary validation."""

    owner: BasePolicy
    observation: dict[str, Any]
    options: dict[str, Any] | None

    def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.owner.strict:
            self.owner.check_observation(self.observation)

        result = self.owner._get_action(self.observation, self.options)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("policy implementations must return an (action, info) tuple")
        action, info = result

        if self.owner.strict:
            self.owner.check_action(action)
        if not isinstance(info, dict):
            raise TypeError("policy info must be a dictionary")
        return action, info


class BasePolicy(ABC):
    """Inference policy contract with validation at the public boundary."""

    def __init__(self, *, strict: bool = True):
        self.strict = bool(strict)

    @abstractmethod
    def check_observation(self, observation: dict[str, Any]) -> None:
        """Reject observations that do not satisfy the implementation contract."""
        raise NotImplementedError

    @abstractmethod
    def check_action(self, action: dict[str, Any]) -> None:
        """Reject malformed actions produced by the implementation."""
        raise NotImplementedError

    @abstractmethod
    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Implementation hook called after optional observation validation."""
        raise NotImplementedError

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _PolicyInvocation(self, observation, options).run()

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Clear implementation-specific temporal state."""
        raise NotImplementedError
