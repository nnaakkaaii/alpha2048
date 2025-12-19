"""Alpha2048 - Deep Reinforcement Learning for 2048 Game."""

__version__ = "0.1.0"
__author__ = "Your Name"

from pkg.environments.game_2048_env import Game2048Env
from pkg.agents.dqn_agent import DQNAgent

__all__ = [
    "Game2048Env",
    "DQNAgent",
]