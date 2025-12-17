"""State encoding utilities."""
import numpy as np
import torch


def encode_board(board: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Encode board state as log2 values.

    Args:
        board: 4x4 numpy array with tile values (0, 2, 4, 8, ...)
        device: torch device

    Returns:
        Tensor of shape (1, 16) with log2 encoded values (normalized)
        Empty cells are 0, tile 2 is 1/17, tile 4 is 2/17, etc.
    """
    # Flatten and convert to log2 (0 stays 0, 2->1, 4->2, etc.)
    flat = board.flatten().astype(np.float32)
    nonzero = flat > 0
    flat[nonzero] = np.log2(flat[nonzero])

    # Normalize by max possible value (2^17 -> 17)
    flat = flat / 17.0

    return torch.from_numpy(flat).unsqueeze(0).to(device)
