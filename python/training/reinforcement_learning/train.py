#!/usr/bin/env python3
"""Train 2048 agent using Double DQN."""
import argparse
import os
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pkg.fields import Action, Game
from pkg.networks import MLP
from pkg.policies import EpsilonGreedy
from pkg.utils import ReplayMemory, Transition, encode_board

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 128
TARGET_UPDATE = 100
MEMORY_SIZE = 100000  # Larger memory for long training
LR = 3e-4  # Slightly higher initial LR for AdamW
WEIGHT_DECAY = 1e-5  # L2 regularization
T_0 = 1000  # Cosine annealing: restart every T_0 episodes
T_MULT = 2  # Each restart doubles the period


def optimize_model(
    memory: ReplayMemory,
    policy_net: MLP,
    target_net: MLP,
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
        # Double DQN: use policy_net to select action, target_net to evaluate
        with torch.no_grad():
            policy_q = policy_net(non_final_next_states)
            target_q = target_net(non_final_next_states)

            non_final_next_actions = [
                a for a in batch.next_actions if a is not None
            ]
            double_q_values = []
            for i, actions in enumerate(non_final_next_actions):
                if actions:
                    # Select best action using policy_net (among legal actions)
                    best_action_idx = policy_q[i, actions].argmax()
                    best_action = actions[best_action_idx]
                    # Evaluate using target_net
                    q_value = target_q[i, best_action]
                    double_q_values.append(q_value)
                else:
                    double_q_values.append(torch.tensor(0.0, device=device))
            next_state_values[non_final_mask] = torch.stack(double_q_values)

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
    policy_net = MLP().to(device)
    target_net = MLP().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts - escapes local optima
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT)

    # Epsilon-greedy with slower decay for long training
    # eps_decay=10000 means ~37% of initial exploration at 10000 steps
    policy = EpsilonGreedy(policy_net, eps_start=1.0, eps_end=0.01, eps_decay=50000)

    memory = ReplayMemory(MEMORY_SIZE)

    # Statistics
    recent_scores = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    steps_done = 0
    start_episode = 0
    best_avg_score = 0

    # Load checkpoint if available (includes optimizer, scheduler state)
    model_path = os.path.join(save_dir, "model.pth")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_net.load_state_dict(checkpoint["model_state"])
        target_net.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        steps_done = checkpoint["steps_done"]
        start_episode = checkpoint["episode"]
        best_avg_score = checkpoint.get("best_avg_score", 0)
        if verbose:
            print(f"Resumed from episode {start_episode}, steps {steps_done}")
    elif os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        policy_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded model from {model_path}")

    game = Game()

    for episode in range(start_episode, num_episodes):
        board = game.reset()
        state = encode_board(board, device)

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

            # Reward shaping: normalize by typical max score
            if score_gained > 0:
                reward = score_gained / 100.0
            else:
                reward = 0.0

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

        # Step scheduler (per episode)
        scheduler.step()

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Calculate current stats
        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0

        # Save checkpoint (with full state for resume)
        if episode % save_interval == 0 and episode > 0:
            # Save best model separately
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                torch.save(policy_net.state_dict(), os.path.join(save_dir, "best_model.pth"))

            # Save checkpoint for resume
            checkpoint = {
                "model_state": policy_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "steps_done": steps_done,
                "episode": episode,
                "best_avg_score": best_avg_score,
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(policy_net.state_dict(), model_path)
            if verbose:
                print(f"Checkpoint saved (episode {episode}, best avg: {best_avg_score:.1f})")

        # Print progress
        if verbose and episode % 100 == 0:
            avg_max = sum(recent_max_tiles) / len(recent_max_tiles) if recent_max_tiles else 0
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            current_lr = scheduler.get_last_lr()[0]
            eps = policy.eps_end + (policy.eps_start - policy.eps_end) * \
                  __import__("math").exp(-steps_done / policy.eps_decay)
            print(
                f"Episode {episode:5d} | "
                f"Avg Score: {avg_score:8.1f} | "
                f"Avg Max Tile: {avg_max:6.1f} | "
                f"Loss: {loss_str} | "
                f"LR: {current_lr:.2e} | "
                f"Eps: {eps:.3f}"
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
