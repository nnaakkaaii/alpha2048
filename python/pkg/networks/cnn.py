"""CNN network for 2048 DQN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pkg.fields import NUM_ACTIONS


class ConvDQN(nn.Module):
    """
    Convolutional Neural Network for 2048.
    
    Input: (batch, 17, 4, 4) - One-Hot encoded tile values
    Output: (batch, 4) - Q-values for each action
    """
    
    def __init__(self, num_tile_types: int = 17, num_actions: int = NUM_ACTIONS):
        super(ConvDQN, self).__init__()
        
        # Conv Layers: Maintain 4x4 spatial size
        self.conv1 = nn.Conv2d(num_tile_types, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # FC Layers
        self.flatten_size = 256 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)