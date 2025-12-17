#!/usr/bin/env python3
"""Train 2048 agent using DQN."""
import argparse
import os
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from pkg.fields import Action, Game
from pkg.networks import CNN
from pkg.policies import EpsilonGreedy
from pkg.utils import ReplayMemory, Transition, encode_board

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 128
TARGET_UPDATE = 100
MEMORY_SIZE = 50000
LR = 1e-4


def optimize_model(
    memory: ReplayMemory,
    policy_net: CNN,
    target_net: CNN,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float | None:
    """Perform one optimization step."""
    if len(memory) < BATCH_SIZE:
        return None

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute mask for non-final states
    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_state], device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for non-final states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    if non_final_mask.sum() > 0:
        # Get legal actions for non-final states
        with torch.no_grad():
            target_q = target_net(non_final_next_states)
            # For each non-final state, get max Q over legal actions
            non_final_next_actions = [
                a for a in batch.next_actions if a is not None
            ]
            max_q_values = []
            for i, actions in enumerate(non_final_next_actions):
                if actions:
                    max_q = target_q[i, actions].max()
                    max_q_values.append(max_q)
                else:
                    max_q_values.append(torch.tensor(0.0, device=device))
            next_state_values[non_final_mask] = torch.stack(max_q_values)

    # Compute expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    # Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def train(
    num_episodes: int,
    save_dir: str,
    device: torch.device,
    verbose: bool = True,
    save_interval: int = 1000,
) -> None:
    """Train the DQN agent."""
    os.makedirs(save_dir, exist_ok=True)

    # Initialize networks
    policy_net = CNN().to(device)
    target_net = CNN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Load existing model if available
    model_path = os.path.join(save_dir, "model.pth")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        policy_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded model from {model_path}")

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    policy = EpsilonGreedy(policy_net)

    # Statistics
    recent_scores = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    steps_done = 0

    game = Game()

    for episode in range(num_episodes):
        board = game.reset()
        state = encode_board(board, device)
        episode_reward = 0

        while True:
            # Get legal actions
            legal_actions = [a.value for a in game.get_legal_actions()]
            if not legal_actions:
                break

            # Select action
            action_idx, _ = policy(state, legal_actions, steps_done)
            action = legal_actions[action_idx]
            steps_done += 1

            # Execute action
            score_gained, done = game.step(Action(action))

            # Reward shaping: use log of score gained
            if score_gained > 0:
                reward = score_gained / 100.0  # Normalize reward
            else:
                reward = 0.0

            episode_reward += score_gained

            # Get next state
            if not done:
                next_board = game.board.copy()
                next_state = encode_board(next_board, device)
                next_legal_actions = [a.value for a in game.get_legal_actions()]
            else:
                next_state = None
                next_legal_actions = None

            # Store transition
            memory.push(
                state,
                torch.tensor([[action]], device=device, dtype=torch.long),
                next_state,
                next_legal_actions,
                torch.tensor([reward], device=device, dtype=torch.float32),
            )

            state = next_state

            if done:
                break

        # Record statistics
        recent_scores.append(game.score)
        recent_max_tiles.append(game.get_max_tile())

        # Optimize model
        loss = optimize_model(memory, policy_net, target_net, optimizer, device)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save model
        if episode % save_interval == 0 and episode > 0:
            torch.save(policy_net.state_dict(), model_path)
            if verbose:
                print(f"Model saved to {model_path}")

        # Print progress
        if verbose and episode % 100 == 0:
            avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
            avg_max = sum(recent_max_tiles) / len(recent_max_tiles) if recent_max_tiles else 0
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(
                f"Episode {episode:5d} | "
                f"Avg Score: {avg_score:8.1f} | "
                f"Avg Max Tile: {avg_max:6.1f} | "
                f"Loss: {loss_str} | "
                f"Memory: {len(memory):5d}"
            )

    # Save final model
    torch.save(policy_net.state_dict(), model_path)
    if verbose:
        print(f"Training complete. Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train 2048 DQN agent")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of training episodes (default: 10000)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save models (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: cpu, cuda, mps, or auto (default: auto)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save model every N episodes (default: 1000)",
    )
    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if not args.quiet:
        print(f"Using device: {device}")

    train(
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        device=device,
        verbose=not args.quiet,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
