"""Base policy class."""
from abc import ABC, abstractmethod

import torch


class BasePolicy(ABC):
    """Abstract base class for action selection policies."""

    @abstractmethod
    def __call__(
        self, state: torch.Tensor, legal_actions: list[int], steps_done: int
    ) -> tuple[int, float]:
        """
        Select an action.

        Args:
            state: Current state tensor
            legal_actions: List of legal action indices
            steps_done: Number of steps taken so far

        Returns:
            Tuple of (selected action index within legal_actions, score)
        """
        pass
