"""Epsilon-greedy policy for exploration."""
import math
import random

import torch

from pkg.policies.base import BasePolicy


class EpsilonGreedy(BasePolicy):
    """Epsilon-greedy policy with decaying epsilon."""

    def __init__(
        self,
        net: torch.nn.Module,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
    ):
        self.net = net
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def __call__(
        self, state: torch.Tensor, legal_actions: list[int], steps_done: int
    ) -> tuple[int, float]:
        """Select action using epsilon-greedy strategy."""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * steps_done / self.eps_decay
        )

        score = 0.0
        if sample > eps_threshold:
            with torch.no_grad():
                q = self.net(state)
                # Get Q-values for legal actions only
                legal_q = q[0, legal_actions]
                value, select_idx = legal_q.max(0)
                score = value.item()
        else:
            select_idx = random.randrange(len(legal_actions))

        return select_idx, score
