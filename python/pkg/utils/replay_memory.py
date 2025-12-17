"""Replay memory for experience replay."""
import random

from pkg.utils.transition import Transition


class ReplayMemory:
    """Fixed-size circular buffer for storing transitions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: list[Transition | None] = []
        self.position = 0

    def push(self, *args) -> None:
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
