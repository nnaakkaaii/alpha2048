# 2048 DQN Design Document

## Overview

Double DQN (Deep Q-Network) implementation for learning to play 2048.

## Network Architecture

### MLP (Multi-Layer Perceptron)

```
Input: (batch, 16) - flattened 4x4 board
    ↓
Linear(16 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU
    ↓
Linear(256 → 4)
    ↓
Output: (batch, 4) - Q-values for [UP, DOWN, LEFT, RIGHT]
```

**Parameters**: ~70,000

**Rationale**: 2048 has a limited state space (4x4 board, ~17 tile values), so a simple MLP is sufficient. CNN would be overkill for this problem.

## State Encoding

```python
def encode_board(board: np.ndarray) -> torch.Tensor:
    # Input: 4x4 board with values [0, 2, 4, 8, ..., 131072]
    # Output: (1, 16) tensor with normalized log2 values

    # Example:
    # [2, 4, 0, 8] → [1, 2, 0, 3] → [1/17, 2/17, 0, 3/17]
```

- Empty cells: 0
- Tile 2: 1/17 (log2(2) = 1)
- Tile 4: 2/17 (log2(4) = 2)
- Tile 2048: 11/17 (log2(2048) = 11)
- Max (2^17): 17/17 = 1.0

## Algorithm: Double DQN

### Standard DQN Problem
- Overestimates Q-values because max operator uses same values for selection and evaluation

### Double DQN Solution
```python
# Standard DQN:
next_q = target_net(next_state).max()

# Double DQN:
best_action = policy_net(next_state).argmax()  # Select with policy_net
next_q = target_net(next_state)[best_action]   # Evaluate with target_net
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GAMMA` | 0.99 | Discount factor |
| `BATCH_SIZE` | 128 | Mini-batch size |
| `TARGET_UPDATE` | 100 | Episodes between target network updates |
| `MEMORY_SIZE` | 100,000 | Replay buffer capacity |
| `LR` | 3e-4 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-5 | L2 regularization |

## Optimizer & Scheduler

### AdamW
- Adam with decoupled weight decay
- Better generalization than standard Adam

### CosineAnnealingWarmRestarts
```
LR Schedule:
    Episode 0-1000:     3e-4 → ~0 (first cycle)
    Episode 1000-3000:  3e-4 → ~0 (second cycle, 2x length)
    Episode 3000-7000:  3e-4 → ~0 (third cycle, 4x length)
    ...
```

**Purpose**: Periodic LR increases help escape local optima during long training.

## Exploration: Epsilon-Greedy

```python
epsilon = eps_end + (eps_start - eps_end) * exp(-steps / eps_decay)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `eps_start` | 1.0 | Initial exploration rate |
| `eps_end` | 0.01 | Minimum exploration rate |
| `eps_decay` | 50,000 | Decay rate (steps) |

**Decay curve**:
- Step 0: ε = 1.0 (100% random)
- Step 50,000: ε ≈ 0.37 (37% random)
- Step 100,000: ε ≈ 0.14 (14% random)
- Step 200,000: ε ≈ 0.02 (2% random)

## Reward Shaping

```python
reward = score_gained / 100.0
```

- Merge 2+2=4: reward = 0.04
- Merge 4+4=8: reward = 0.08
- Merge 1024+1024=2048: reward = 20.48

## Training Loop

```
for episode in range(num_episodes):
    1. Reset game
    2. While not game_over:
        a. Encode state
        b. Select action (epsilon-greedy)
        c. Execute action, get reward
        d. Store transition in replay buffer
    3. Sample batch from replay buffer
    4. Compute Double DQN loss
    5. Update policy_net (AdamW + gradient clipping)
    6. Step scheduler
    7. Periodically update target_net
    8. Save checkpoint
```

## File Structure

```
python/
├── pkg/
│   ├── networks/
│   │   └── mlp.py              # MLP Q-network
│   ├── policies/
│   │   ├── epsilon_greedy.py   # Exploration policy
│   │   └── greedy.py           # Evaluation policy
│   └── utils/
│       ├── replay_memory.py    # Experience replay
│       ├── state.py            # Board encoding
│       └── transition.py       # (s, a, s', r) tuple
└── training/
    └── reinforcement_learning/
        ├── train.py            # Training loop
        └── test.py             # Evaluation
```

## Checkpoints

| File | Content |
|------|---------|
| `model.pth` | Latest model weights |
| `best_model.pth` | Best average score model |
| `checkpoint.pth` | Full state (model, optimizer, scheduler, episode, steps) |

## Usage

```bash
# Train
.venv/bin/train --episodes 100000 --save-interval 1000

# Resume training (automatic from checkpoint)
.venv/bin/train --episodes 100000

# Evaluate
.venv/bin/test-model --games 100
```

## Performance Baseline

| Policy | Avg Score | 512 Rate |
|--------|-----------|----------|
| Random | ~1,100 | ~0% |
| DQN (1000 ep) | ~2,300 | ~8% |
