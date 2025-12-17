"""Tests for reinforcement learning components."""
import numpy as np
import pytest
import torch

from pkg.fields import Action, Game
from pkg.networks import CNN
from pkg.policies import EpsilonGreedy, Greedy
from pkg.utils import ReplayMemory, Transition, encode_board


class TestCNN:
    """Test CNN network."""

    def test_output_shape(self):
        """Test CNN output shape."""
        net = CNN()
        x = torch.randn(1, 16, 4, 4)
        out = net(x)
        assert out.shape == (1, 4)

    def test_batch_output_shape(self):
        """Test CNN with batch input."""
        net = CNN()
        x = torch.randn(32, 16, 4, 4)
        out = net(x)
        assert out.shape == (32, 4)


class TestEncodeBoard:
    """Test board encoding."""

    def test_empty_board(self):
        """Test encoding empty board."""
        board = np.zeros((4, 4), dtype=np.int32)
        device = torch.device("cpu")
        encoded = encode_board(board, device)
        assert encoded.shape == (1, 16, 4, 4)
        # All tiles are 0, so channel 0 should be all 1s
        assert encoded[0, 0].sum() == 16

    def test_single_tile(self):
        """Test encoding single tile."""
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = 2  # 2 = 2^1
        device = torch.device("cpu")
        encoded = encode_board(board, device)
        # Channel 1 should have a 1 at position (0, 0)
        assert encoded[0, 1, 0, 0] == 1
        # Channel 0 should have 15 ones (all zeros except the 2 tile)
        assert encoded[0, 0].sum() == 15

    def test_multiple_tiles(self):
        """Test encoding multiple tiles."""
        board = np.array([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int32)
        device = torch.device("cpu")
        encoded = encode_board(board, device)
        # Check each tile
        assert encoded[0, 1, 0, 0] == 1  # 2 = 2^1
        assert encoded[0, 2, 0, 1] == 1  # 4 = 2^2
        assert encoded[0, 3, 0, 2] == 1  # 8 = 2^3
        assert encoded[0, 4, 0, 3] == 1  # 16 = 2^4


class TestReplayMemory:
    """Test replay memory."""

    def test_push_and_sample(self):
        """Test pushing and sampling transitions."""
        memory = ReplayMemory(100)
        for i in range(50):
            memory.push(
                torch.randn(1, 16, 4, 4),
                torch.tensor([[i % 4]]),
                torch.randn(1, 16, 4, 4),
                [0, 1, 2, 3],
                torch.tensor([1.0]),
            )
        assert len(memory) == 50
        samples = memory.sample(10)
        assert len(samples) == 10

    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        memory = ReplayMemory(10)
        for i in range(20):
            memory.push(
                torch.randn(1, 16, 4, 4),
                torch.tensor([[0]]),
                None,
                None,
                torch.tensor([float(i)]),
            )
        assert len(memory) == 10
        # Check that only recent values are in memory
        rewards = [t.reward.item() for t in memory.memory]
        assert min(rewards) >= 10


class TestEpsilonGreedy:
    """Test epsilon-greedy policy."""

    def test_action_selection(self):
        """Test action selection."""
        net = CNN()
        policy = EpsilonGreedy(net, eps_start=0.0, eps_end=0.0, eps_decay=1)
        state = torch.randn(1, 16, 4, 4)
        legal_actions = [0, 1, 2, 3]
        action_idx, score = policy(state, legal_actions, 1000)
        assert 0 <= action_idx < len(legal_actions)
        assert isinstance(score, float)

    def test_exploration_at_start(self):
        """Test high exploration at start."""
        net = CNN()
        policy = EpsilonGreedy(net, eps_start=1.0, eps_end=0.0, eps_decay=1000)
        state = torch.randn(1, 16, 4, 4)
        legal_actions = [0, 1, 2, 3]
        # With eps=1.0, should be random (can't test randomness directly)
        action_idx, _ = policy(state, legal_actions, 0)
        assert 0 <= action_idx < len(legal_actions)


class TestGreedy:
    """Test greedy policy."""

    def test_greedy_selection(self):
        """Test greedy action selection."""
        net = CNN()
        policy = Greedy(net)
        state = torch.randn(1, 16, 4, 4)
        legal_actions = [0, 1, 2, 3]
        action_idx, score = policy(state, legal_actions, 0)
        assert 0 <= action_idx < len(legal_actions)
        assert isinstance(score, float)


class TestIntegration:
    """Integration tests for RL training loop."""

    def test_single_episode(self):
        """Test running a single episode."""
        device = torch.device("cpu")
        net = CNN().to(device)
        policy = EpsilonGreedy(net)
        game = Game()

        board = game.reset()
        state = encode_board(board, device)
        steps = 0

        while not game.is_game_over() and steps < 1000:
            legal_actions = [a.value for a in game.get_legal_actions()]
            if not legal_actions:
                break

            action_idx, _ = policy(state, legal_actions, steps)
            action = legal_actions[action_idx]
            _, done = game.step(Action(action))
            steps += 1

            if not done:
                state = encode_board(game.board.copy(), device)
            else:
                break

        assert steps > 0
        assert game.score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
