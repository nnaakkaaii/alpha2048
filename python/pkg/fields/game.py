"""2048 game logic."""
import random
from typing import Optional

import numpy as np

from pkg.fields.board import NUM_COLUMNS, NUM_ROWS
from pkg.fields.actions import Action


class Game:
    """2048 game implementation."""

    def __init__(self):
        self.board: np.ndarray = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.int32)
        self.score: int = 0
        self._game_over: bool = False

    def reset(self) -> np.ndarray:
        """Reset the game and return initial state."""
        self.board = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.int32)
        self.score = 0
        self._game_over = False
        self._spawn_tile()
        self._spawn_tile()
        return self.board.copy()

    def _get_empty_cells(self) -> list[tuple[int, int]]:
        """Get list of empty cell coordinates."""
        return list(zip(*np.where(self.board == 0)))

    def _spawn_tile(self) -> bool:
        """Spawn a new tile (2 with 90% prob, 4 with 10% prob)."""
        empty_cells = self._get_empty_cells()
        if not empty_cells:
            return False
        row, col = random.choice(empty_cells)
        self.board[row, col] = 2 if random.random() < 0.9 else 4
        return True

    def _compress(self, row: np.ndarray) -> tuple[np.ndarray, int]:
        """Compress a row by sliding non-zero tiles to the left and merging."""
        # Remove zeros
        non_zero = row[row != 0]
        merged = []
        score = 0
        skip = False

        for i, val in enumerate(non_zero):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = val * 2
                merged.append(merged_val)
                score += merged_val
                skip = True
            else:
                merged.append(val)

        # Pad with zeros
        result = np.zeros(len(row), dtype=np.int32)
        result[: len(merged)] = merged
        return result, score

    def _move_left(self) -> tuple[bool, int]:
        """Move all tiles left."""
        moved = False
        total_score = 0
        new_board = np.zeros_like(self.board)

        for i in range(NUM_ROWS):
            new_row, score = self._compress(self.board[i])
            if not np.array_equal(self.board[i], new_row):
                moved = True
            new_board[i] = new_row
            total_score += score

        self.board = new_board
        return moved, total_score

    def _move_right(self) -> tuple[bool, int]:
        """Move all tiles right."""
        moved = False
        total_score = 0
        new_board = np.zeros_like(self.board)

        for i in range(NUM_ROWS):
            new_row, score = self._compress(self.board[i][::-1])
            new_board[i] = new_row[::-1]
            if not np.array_equal(self.board[i], new_board[i]):
                moved = True
            total_score += score

        self.board = new_board
        return moved, total_score

    def _move_up(self) -> tuple[bool, int]:
        """Move all tiles up."""
        moved = False
        total_score = 0
        new_board = np.zeros_like(self.board)

        for j in range(NUM_COLUMNS):
            col = self.board[:, j]
            new_col, score = self._compress(col)
            new_board[:, j] = new_col
            if not np.array_equal(col, new_col):
                moved = True
            total_score += score

        self.board = new_board
        return moved, total_score

    def _move_down(self) -> tuple[bool, int]:
        """Move all tiles down."""
        moved = False
        total_score = 0
        new_board = np.zeros_like(self.board)

        for j in range(NUM_COLUMNS):
            col = self.board[:, j]
            new_col, score = self._compress(col[::-1])
            new_board[:, j] = new_col[::-1]
            if not np.array_equal(col, new_board[:, j]):
                moved = True
            total_score += score

        self.board = new_board
        return moved, total_score

    def step(self, action: Action) -> tuple[int, bool]:
        """
        Execute an action.

        Args:
            action: The action to execute (UP, DOWN, LEFT, RIGHT)

        Returns:
            tuple of (score_gained, game_over)
        """
        if self._game_over:
            return 0, True

        if action == Action.UP:
            moved, score = self._move_up()
        elif action == Action.DOWN:
            moved, score = self._move_down()
        elif action == Action.LEFT:
            moved, score = self._move_left()
        elif action == Action.RIGHT:
            moved, score = self._move_right()
        else:
            raise ValueError(f"Invalid action: {action}")

        if moved:
            self.score += score
            self._spawn_tile()

        # Check game over
        self._game_over = self._check_game_over()

        return score, self._game_over

    def _check_game_over(self) -> bool:
        """Check if no more moves are possible."""
        # If there are empty cells, game is not over
        if 0 in self.board:
            return False

        # Check for possible merges horizontally
        for i in range(NUM_ROWS):
            for j in range(NUM_COLUMNS - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False

        # Check for possible merges vertically
        for i in range(NUM_ROWS - 1):
            for j in range(NUM_COLUMNS):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False

        return True

    def is_game_over(self) -> bool:
        """Return whether the game is over."""
        return self._game_over

    def get_legal_actions(self) -> list[Action]:
        """Get list of legal actions (actions that would change the board)."""
        legal = []
        original_board = self.board.copy()
        original_score = self.score

        for action in Action:
            self.board = original_board.copy()
            if action == Action.UP:
                moved, _ = self._move_up()
            elif action == Action.DOWN:
                moved, _ = self._move_down()
            elif action == Action.LEFT:
                moved, _ = self._move_left()
            elif action == Action.RIGHT:
                moved, _ = self._move_right()

            if moved:
                legal.append(action)

        # Restore state
        self.board = original_board
        self.score = original_score
        return legal

    def get_max_tile(self) -> int:
        """Return the maximum tile value on the board."""
        return int(np.max(self.board))

    def render(self) -> str:
        """Render the board as a string."""
        lines = []
        lines.append("+" + "------+" * NUM_COLUMNS)
        for row in self.board:
            line = "|"
            for val in row:
                if val == 0:
                    line += "      |"
                else:
                    line += f"{val:^6}|"
            lines.append(line)
            lines.append("+" + "------+" * NUM_COLUMNS)
        lines.append(f"Score: {self.score}")
        return "\n".join(lines)

    def copy(self) -> "Game":
        """Create a copy of the game."""
        new_game = Game()
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game._game_over = self._game_over
        return new_game
