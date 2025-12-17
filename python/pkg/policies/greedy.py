"""Greedy policy for evaluation."""
import torch

from pkg.policies.base import BasePolicy


class Greedy(BasePolicy):
    """Greedy policy - always select the best action."""

    def __init__(self, net: torch.nn.Module):
        self.net = net

    def __call__(
        self, state: torch.Tensor, legal_actions: list[int], steps_done: int
    ) -> tuple[int, float]:
        """Select best action according to Q-values."""
        with torch.no_grad():
            q = self.net(state)
            legal_q = q[0, legal_actions]
            value, select_idx = legal_q.max(0)
            score = value.item()

        return select_idx, score
