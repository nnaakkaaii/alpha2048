"""2048 Gym Environment."""
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pkg.fields import Action, NUM_ACTIONS, NUM_COLUMNS, NUM_ROWS, Game


class Game2048Env(gym.Env):
    """
    OpenAI Gym environment for 2048 game.

    Observation:
        4x4 board with tile values (0 for empty, powers of 2 for tiles)

    Actions:
        0: UP
        1: DOWN
        2: LEFT
        3: RIGHT

    Reward:
        Score gained from merging tiles in each step.
        Can be configured to use different reward schemes.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.game = Game()
        self.render_mode = render_mode

        # Observation space: 4x4 board
        # Using large upper bound for tile values (max observed is 131072 = 2^17)
        self.observation_space = spaces.Box(
            low=0,
            high=2**17,
            shape=(NUM_ROWS, NUM_COLUMNS),
            dtype=np.int32,
        )

        # Action space: 4 directions
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        observation = self.game.reset()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human()

        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)

        Returns:
            observation: Current board state
            reward: Score gained from this action
            terminated: Whether the game is over
            truncated: Always False (no time limit)
            info: Additional information
        """
        action_enum = Action(action)
        score_gained, game_over = self.game.step(action_enum)

        observation = self.game.board.copy()
        reward = float(score_gained)
        terminated = game_over
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human()

        return observation, reward, terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the current state."""
        return {
            "score": self.game.score,
            "max_tile": self.game.get_max_tile(),
            "legal_actions": [a.value for a in self.game.get_legal_actions()],
        }

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self.game.render()
        elif self.render_mode == "human":
            self._render_human()
            return None
        return None

    def _render_human(self) -> None:
        """Render to console for human viewing."""
        print("\033[2J\033[H")  # Clear screen
        print(self.game.render())

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        return [a.value for a in self.game.get_legal_actions()]

    def close(self) -> None:
        """Clean up resources."""
        pass
