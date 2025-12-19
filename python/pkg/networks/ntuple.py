"""N-tuple network for 2048 with TD learning.

Based on state-of-the-art research:
- "Temporal Difference Learning of N-Tuple Networks for the Game 2048" by Wu et al.
- "Multi-Stage Temporal Difference Learning for 2048" by Szubert & Jaskowski

This implementation uses:
- 4-tuple patterns (rows, columns, 2x2 squares)
- 6-tuple patterns (L-shapes, rectangles)
- 8-fold symmetry (4 rotations x 2 reflections)
"""
import os
import pickle
from typing import Optional

import numpy as np

from pkg.fields.board import NUM_ROWS, NUM_COLUMNS


# Maximum tile value we track (2^15 = 32768)
MAX_TILE_POWER = 16  # 0-15, where 0=empty, 1=2, 2=4, ..., 15=32768


def tile_to_index(tile: int) -> int:
    """Convert tile value to index (0=empty, 1=2, 2=4, ...)."""
    if tile == 0:
        return 0
    # tile = 2^k, so k = log2(tile)
    return int(np.log2(tile))


def board_to_indices(board: np.ndarray) -> np.ndarray:
    """Convert board values to indices."""
    indices = np.zeros_like(board, dtype=np.int32)
    mask = board > 0
    indices[mask] = np.log2(board[mask]).astype(np.int32)
    return indices


class NTupleNetwork:
    """N-tuple network for 2048 value function approximation.

    Uses multiple N-tuple patterns with 8-fold symmetry to evaluate board states.
    Each pattern has a Look-Up Table (LUT) that maps tile configurations to values.
    """

    def __init__(self, learning_rate: float = 0.1):
        """Initialize N-tuple network.

        Args:
            learning_rate: Learning rate for TD updates
        """
        self.learning_rate = learning_rate

        # Define N-tuple patterns (row, col) coordinates
        # 4-tuple patterns: horizontal lines
        self.patterns_4_horizontal = [
            [(0, 0), (0, 1), (0, 2), (0, 3)],  # Row 0
            [(1, 0), (1, 1), (1, 2), (1, 3)],  # Row 1
            [(2, 0), (2, 1), (2, 2), (2, 3)],  # Row 2
            [(3, 0), (3, 1), (3, 2), (3, 3)],  # Row 3
        ]

        # 4-tuple patterns: vertical lines
        self.patterns_4_vertical = [
            [(0, 0), (1, 0), (2, 0), (3, 0)],  # Col 0
            [(0, 1), (1, 1), (2, 1), (3, 1)],  # Col 1
            [(0, 2), (1, 2), (2, 2), (3, 2)],  # Col 2
            [(0, 3), (1, 3), (2, 3), (3, 3)],  # Col 3
        ]

        # 4-tuple patterns: 2x2 squares
        self.patterns_4_square = [
            [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
            for i in range(3) for j in range(3)
        ]

        # 6-tuple patterns: L-shapes and rectangles
        self.patterns_6 = [
            # L-shapes
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],  # 2x3 rectangle
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],  # 3x2 rectangle
            [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)],  # L-shape 1
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 1), (2, 2)],  # L-shape 2
            # Axe patterns
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)],  # Axe 1
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],  # Axe 2
            [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3)],  # Snake
            [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],  # Diagonal snake
        ]

        # Combine all patterns
        self.all_4_patterns = (
            self.patterns_4_horizontal +
            self.patterns_4_vertical +
            self.patterns_4_square
        )
        self.all_6_patterns = self.patterns_6

        # Create LUTs for each pattern type
        # 4-tuple: 16^4 = 65536 entries
        # 6-tuple: 16^6 = 16777216 entries
        self.lut_4 = [np.zeros(MAX_TILE_POWER ** 4, dtype=np.float32)
                      for _ in self.all_4_patterns]
        self.lut_6 = [np.zeros(MAX_TILE_POWER ** 6, dtype=np.float32)
                      for _ in self.all_6_patterns]

        # Precompute symmetry transformations
        self._init_symmetry_patterns()

        # Initialize vectorized index arrays
        self._init_index_arrays()

    def _init_symmetry_patterns(self):
        """Precompute all symmetry-transformed patterns."""
        self.symmetric_4_patterns = []
        self.symmetric_6_patterns = []

        for pattern in self.all_4_patterns:
            symmetric = self._get_symmetric_patterns(pattern)
            self.symmetric_4_patterns.append(symmetric)

        for pattern in self.all_6_patterns:
            symmetric = self._get_symmetric_patterns(pattern)
            self.symmetric_6_patterns.append(symmetric)

    def _get_symmetric_patterns(self, pattern: list) -> list:
        """Get all 8 symmetric versions of a pattern.

        Applies 4 rotations x 2 reflections = 8 transformations.
        """
        symmetric = []
        coords = np.array(pattern, dtype=np.int32)

        for rotation in range(4):
            # Rotate 90 degrees clockwise: (r, c) -> (c, 3-r)
            rotated = coords.copy()
            for _ in range(rotation):
                rotated = np.column_stack([
                    rotated[:, 1],
                    3 - rotated[:, 0]
                ])
            symmetric.append(rotated.tolist())

            # Reflect horizontally: (r, c) -> (r, 3-c)
            reflected = rotated.copy()
            reflected[:, 1] = 3 - reflected[:, 1]
            symmetric.append(reflected.tolist())

        return symmetric

    def _init_index_arrays(self):
        """Precompute index multipliers for vectorized operations."""
        # Multipliers for 4-tuple: [1, 16, 256, 4096]
        self.mult_4 = np.array([MAX_TILE_POWER ** i for i in range(4)], dtype=np.int64)
        # Multipliers for 6-tuple: [1, 16, 256, 4096, 65536, 1048576]
        self.mult_6 = np.array([MAX_TILE_POWER ** i for i in range(6)], dtype=np.int64)

        # Convert pattern lists to numpy arrays for faster indexing
        # Shape: (num_patterns, 8_symmetries, tuple_size, 2)
        self.pattern_4_arr = np.array(self.symmetric_4_patterns, dtype=np.int32)
        self.pattern_6_arr = np.array(self.symmetric_6_patterns, dtype=np.int32)

    def _compute_indices_4(self, board_indices: np.ndarray) -> np.ndarray:
        """Compute all 4-tuple indices vectorized."""
        # pattern_4_arr shape: (num_patterns, 8, 4, 2)
        # Extract row and col indices
        rows = self.pattern_4_arr[:, :, :, 0]  # (num_patterns, 8, 4)
        cols = self.pattern_4_arr[:, :, :, 1]  # (num_patterns, 8, 4)

        # Get tile indices at each position
        tile_indices = board_indices[rows, cols]  # (num_patterns, 8, 4)

        # Compute LUT indices: sum of tile_idx * multiplier
        lut_indices = np.sum(tile_indices * self.mult_4, axis=2)  # (num_patterns, 8)
        return lut_indices.astype(np.int64)

    def _compute_indices_6(self, board_indices: np.ndarray) -> np.ndarray:
        """Compute all 6-tuple indices vectorized."""
        rows = self.pattern_6_arr[:, :, :, 0]
        cols = self.pattern_6_arr[:, :, :, 1]
        tile_indices = board_indices[rows, cols]
        lut_indices = np.sum(tile_indices * self.mult_6, axis=2)
        return lut_indices.astype(np.int64)

    def evaluate(self, board: np.ndarray) -> float:
        """Evaluate board state value.

        Args:
            board: 4x4 board with tile values (0, 2, 4, 8, ...)

        Returns:
            Estimated value of the state
        """
        board_indices = board_to_indices(board)

        # Compute all indices at once
        indices_4 = self._compute_indices_4(board_indices)  # (num_patterns, 8)
        indices_6 = self._compute_indices_6(board_indices)  # (num_patterns, 8)

        # Sum values from all LUTs
        value = 0.0
        for pattern_idx in range(len(self.lut_4)):
            value += np.sum(self.lut_4[pattern_idx][indices_4[pattern_idx]])
        for pattern_idx in range(len(self.lut_6)):
            value += np.sum(self.lut_6[pattern_idx][indices_6[pattern_idx]])

        return float(value)

    def update(self, board: np.ndarray, delta: float) -> None:
        """Update LUT weights using TD error.

        Args:
            board: 4x4 board state
            delta: TD error (target - prediction)
        """
        board_indices = board_to_indices(board)

        # Count total number of feature updates
        num_4_features = len(self.symmetric_4_patterns) * 8
        num_6_features = len(self.symmetric_6_patterns) * 8
        total_features = num_4_features + num_6_features

        # Update amount per feature
        update = delta * self.learning_rate / total_features

        # Compute all indices at once
        indices_4 = self._compute_indices_4(board_indices)
        indices_6 = self._compute_indices_6(board_indices)

        # Update 4-tuple LUTs
        for pattern_idx in range(len(self.lut_4)):
            for sym_idx in range(8):
                self.lut_4[pattern_idx][indices_4[pattern_idx, sym_idx]] += update

        # Update 6-tuple LUTs
        for pattern_idx in range(len(self.lut_6)):
            for sym_idx in range(8):
                self.lut_6[pattern_idx][indices_6[pattern_idx, sym_idx]] += update

    def get_best_action(self, game, legal_actions: list) -> tuple[int, float]:
        """Select best action using afterstate evaluation.

        Args:
            game: Game instance
            legal_actions: List of legal action values

        Returns:
            (best_action, best_value)
        """
        best_action = legal_actions[0]
        best_value = float('-inf')

        for action in legal_actions:
            # Simulate action to get afterstate
            afterstate, reward = self._get_afterstate(game, action)
            if afterstate is not None:
                value = reward + self.evaluate(afterstate)
                if value > best_value:
                    best_value = value
                    best_action = action

        return best_action, best_value

    def _get_afterstate(self, game, action: int) -> tuple[Optional[np.ndarray], int]:
        """Get afterstate (state after action, before random tile).

        Returns:
            (afterstate_board, reward) or (None, 0) if action is invalid
        """
        from pkg.fields import Action

        # Save original state
        original_board = game.board.copy()
        original_score = game.score

        # Execute action (this modifies game state)
        game.board = original_board.copy()

        if action == Action.UP.value:
            moved, reward = game._move_up()
        elif action == Action.DOWN.value:
            moved, reward = game._move_down()
        elif action == Action.LEFT.value:
            moved, reward = game._move_left()
        elif action == Action.RIGHT.value:
            moved, reward = game._move_right()
        else:
            game.board = original_board
            game.score = original_score
            return None, 0

        if not moved:
            game.board = original_board
            game.score = original_score
            return None, 0

        # Get afterstate (before random tile spawn)
        afterstate = game.board.copy()

        # Restore original state
        game.board = original_board
        game.score = original_score

        return afterstate, reward

    def save(self, filepath: str) -> None:
        """Save network weights to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        data = {
            'lut_4': self.lut_4,
            'lut_6': self.lut_6,
            'learning_rate': self.learning_rate,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load network weights from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.lut_4 = data['lut_4']
        self.lut_6 = data['lut_6']
        self.learning_rate = data.get('learning_rate', self.learning_rate)

    def get_memory_usage_mb(self) -> float:
        """Get approximate memory usage in MB."""
        lut_4_size = sum(lut.nbytes for lut in self.lut_4)
        lut_6_size = sum(lut.nbytes for lut in self.lut_6)
        return (lut_4_size + lut_6_size) / (1024 * 1024)
