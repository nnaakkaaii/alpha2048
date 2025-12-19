"""MLP network for 2048 DQN."""
import torch.nn as nn

from pkg.fields import NUM_ACTIONS, NUM_SQUARES


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for 2048.

    Input: (batch, 16) - log2 encoded tile values
    Output: (batch, 4) - Q-values for each action
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NUM_SQUARES, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.fc(x)
