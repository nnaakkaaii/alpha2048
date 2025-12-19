# Alpha2048 - Deep Reinforcement Learning for 2048

A Python implementation of Deep Q-Network (DQN) and other reinforcement learning algorithms for playing the 2048 game.

## Features

- Deep Q-Network (DQN) with experience replay
- Double DQN for improved stability
- CNN-based state representation
- Prioritized experience replay
- Dueling network architecture
- AdamW optimizer with cosine annealing scheduler

## Installation

### From GitHub (recommended for Google Colab)

```bash
pip install git+https://github.com/nnaakkaaii/alpha2048.git#subdirectory=python
```

### For development

```bash
git clone https://github.com/nnaakkaaii/alpha2048.git
cd alpha2048/python
pip install -e .
```

## Usage in Google Colab

```python
# Install the package
!pip install git+https://github.com/nnaakkaaii/alpha2048.git#subdirectory=python

# Import and use
from pkg.environments.game_2048_env import Game2048Env
from pkg.networks.dqn import DQN
from pkg.agents.dqn_agent import DQNAgent
from training.reinforcement_learning.train import train_dqn

# Create environment
env = Game2048Env()

# Train the agent
agent = train_dqn(
    episodes=1000,
    batch_size=32,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Test the trained agent
from training.reinforcement_learning.test import test_dqn
test_dqn(agent, num_games=10)
```

## Training

### Command Line

```bash
# Train with default settings
alpha2048-train --episodes 10000 --batch-size 64

# Test trained model
alpha2048-test --model-path model.pth --num-games 100
```

### Python Script

```python
from training.reinforcement_learning.train import train_dqn

agent = train_dqn(
    episodes=10000,
    batch_size=64,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    target_update=10,
    memory_size=10000,
    save_path='models/',
    device='cuda'
)
```

## Architecture

The project includes several network architectures:

- **MLP**: Simple Multi-Layer Perceptron
- **CNN**: Convolutional Neural Network for spatial feature extraction
- **DQN**: Deep Q-Network with experience replay
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separate value and advantage streams

## Project Structure

```
python/
├── pkg/
│   ├── agents/          # RL agents (DQN, etc.)
│   ├── environments/    # 2048 game environment
│   ├── networks/        # Neural network architectures
│   └── utils/           # Utilities (state processing, memory)
├── training/
│   └── reinforcement_learning/
│       ├── train.py     # Training script
│       └── test.py      # Testing script
├── setup.py             # Package setup
└── requirements.txt     # Dependencies
```

## License

MIT License