"""Tests for 2048 environment."""
import numpy as np
import pytest

from pkg.envs import Game2048Env
from pkg.fields import Action, Game


class TestGame:
    """Test 2048 game logic."""

    def test_reset(self):
        """Test game reset creates valid initial state."""
        game = Game()
        board = game.reset()

        assert board.shape == (4, 4)
        # Should have exactly 2 tiles
        assert np.count_nonzero(board) == 2
        # Tiles should be 2 or 4
        for val in board.flatten():
            if val != 0:
                assert val in (2, 4)

    def test_move_left_simple(self):
        """Test simple left move."""
        game = Game()
        game.board = np.array([
            [0, 0, 0, 2],
            [0, 0, 2, 0],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
        ], dtype=np.int32)

        # Manually move left without spawning
        moved, score = game._move_left()

        assert moved is True
        assert score == 0
        # First column should have all 2s
        assert game.board[0, 0] == 2
        assert game.board[1, 0] == 2
        assert game.board[2, 0] == 2
        assert game.board[3, 0] == 2

    def test_merge_tiles(self):
        """Test tile merging."""
        game = Game()
        game.board = np.array([
            [2, 2, 0, 0],
            [4, 4, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)

        moved, score = game._move_left()

        assert moved is True
        assert score == 4 + 8 + 8  # 2+2=4, 4+4=8, 4+4=8
        assert game.board[0, 0] == 4
        assert game.board[0, 1] == 0
        assert game.board[1, 0] == 8
        assert game.board[1, 1] == 8

    def test_game_over_detection(self):
        """Test game over detection."""
        game = Game()
        # Create a board where no moves are possible
        game.board = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ], dtype=np.int32)

        assert game._check_game_over() is True

    def test_game_not_over_with_empty(self):
        """Test game not over when empty cells exist."""
        game = Game()
        game.board = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 0],  # One empty cell
        ], dtype=np.int32)

        assert game._check_game_over() is False

    def test_game_not_over_with_merge_possible(self):
        """Test game not over when merge is possible."""
        game = Game()
        game.board = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 4],  # Can merge the two 4s
        ], dtype=np.int32)

        assert game._check_game_over() is False

    def test_legal_actions(self):
        """Test getting legal actions."""
        game = Game()
        game.board = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)

        legal = game.get_legal_actions()
        # DOWN, RIGHT should be legal (move tile)
        # UP, LEFT should be illegal (no change)
        assert Action.DOWN in legal
        assert Action.RIGHT in legal
        assert Action.UP not in legal
        assert Action.LEFT not in legal


class TestGame2048Env:
    """Test 2048 Gym environment."""

    def test_env_creation(self):
        """Test environment can be created."""
        env = Game2048Env()
        assert env.observation_space.shape == (4, 4)
        assert env.action_space.n == 4

    def test_env_reset(self):
        """Test environment reset."""
        env = Game2048Env()
        obs, info = env.reset(seed=42)

        assert obs.shape == (4, 4)
        assert "score" in info
        assert "max_tile" in info
        assert "legal_actions" in info

    def test_env_step(self):
        """Test environment step."""
        env = Game2048Env()
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(0)  # UP

        assert obs.shape == (4, 4)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert truncated is False

    def test_env_render_ansi(self):
        """Test ANSI rendering."""
        env = Game2048Env(render_mode="ansi")
        env.reset(seed=42)

        output = env.render()
        assert isinstance(output, str)
        assert "Score:" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
