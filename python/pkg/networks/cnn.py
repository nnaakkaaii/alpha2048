"""CNN network for 2048 DQN."""
import torch.nn as nn

from pkg.fields import NUM_ACTIONS, NUM_COLUMNS, NUM_ROWS


class CNN(nn.Module):
    """
    Convolutional Neural Network for 2048.

    Input: (batch, 16, 4, 4) - one-hot encoded tile values (0-15 for 2^0 to 2^15)
    Output: (batch, 4) - Q-values for each action
    """

    def __init__(self, num_channels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        # After conv: (batch, num_channels, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(num_channels * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
