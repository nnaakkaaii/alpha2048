"""State encoding utilities."""
import numpy as np
import torch


def encode_board(board: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Encode board state as one-hot tensor.

    Args:
        board: 4x4 numpy array with tile values (0, 2, 4, 8, ...)
        device: torch device

    Returns:
        Tensor of shape (1, 16, 4, 4) with one-hot encoding
        Channel i represents tiles with value 2^i (channel 0 = empty)
    """
    # Convert tile values to log2 indices (0 stays 0, 2->1, 4->2, etc.)
    board_log = np.zeros_like(board, dtype=np.int64)
    nonzero = board > 0
    board_log[nonzero] = np.log2(board[nonzero]).astype(np.int64)

    # One-hot encode (16 channels for values 0 to 2^15)
    one_hot = np.zeros((16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            one_hot[board_log[i, j], i, j] = 1.0

    return torch.from_numpy(one_hot).unsqueeze(0).to(device)
