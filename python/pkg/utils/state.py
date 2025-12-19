"""State encoding utilities."""
import numpy as np
import torch
import torch.nn.functional as F

# Mapping: 0->0, 2->1, 4->2, ..., 65536->16
TILE_MAP = {0: 0}
for i in range(1, 17):
    TILE_MAP[2**i] = i


def encode_board(board: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Encode board state as One-Hot encoding.

    Args:
        board: 4x4 numpy array with tile values (0, 2, 4, 8, ...)
        device: torch device

    Returns:
        Tensor of shape (1, 17, 4, 4) with One-Hot encoded values
        17 channels represent tiles from 0 to 2^16 (65536)
    """
    # 1. Map raw values to indices
    board_indices = np.vectorize(TILE_MAP.get)(board)
    
    # 2. To Tensor: (1, 4, 4)
    tensor_indices = torch.LongTensor(board_indices).unsqueeze(0).to(device)
    
    # 3. One-Hot: (1, 4, 4, 17)
    one_hot = F.one_hot(tensor_indices, num_classes=17)
    
    # 4. Permute to (1, 17, 4, 4) for CNN
    return one_hot.permute(0, 3, 1, 2).float()
