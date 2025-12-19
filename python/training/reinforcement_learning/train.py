#!/usr/bin/env python3
"""Train 2048 agent using Double DQN or N-tuple + TD learning."""
import argparse
import math
import os
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from pkg.fields import Action, Game
from pkg.networks import ConvDQN, NTupleNetwork
from pkg.policies import EpsilonGreedy
from pkg.utils import ReplayMemory, Transition, encode_board

# DQN Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 512
TARGET_UPDATE = 100
MEMORY_SIZE = 1000000
LR = 3e-4
WEIGHT_DECAY = 1e-5
T_0 = 1000
T_MULT = 2

# N-tuple TD Hyperparameters
NTUPLE_LR = 0.1  # Learning rate for N-tuple TD
NTUPLE_LR_DECAY = 0.99999  # Per-step decay


def optimize_model(
    memory: ReplayMemory,
    policy_net: ConvDQN,
    target_net: ConvDQN,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
) -> float | None:
    """Perform one optimization step for DQN."""
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_state], device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)

    if non_final_mask.sum() > 0:
        with torch.no_grad():
            policy_q = policy_net(non_final_next_states)
            target_q = target_net(non_final_next_states)

            non_final_next_actions = [
                a for a in batch.next_actions if a is not None
            ]
            double_q_values = []
            for i, actions in enumerate(non_final_next_actions):
                if actions:
                    best_action_idx = policy_q[i, actions].argmax()
                    best_action = actions[best_action_idx]
                    q_value = target_q[i, best_action]
                    double_q_values.append(q_value)
                else:
                    double_q_values.append(torch.tensor(0.0, device=device))
            next_state_values[non_final_mask] = torch.stack(double_q_values)

    expected_state_action_values = reward_batch + (gamma * next_state_values)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def train_dqn(
    num_episodes: int,
    save_dir: str,
    device: torch.device,
    verbose: bool = True,
    save_interval: int = 1000,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    target_update: int = TARGET_UPDATE,
    memory_size: int = MEMORY_SIZE,
    t_0: int = T_0,
    t_mult: int = T_MULT,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: int = 500000,
) -> None:
    """Train the DQN agent with customizable hyperparameters."""
    os.makedirs(save_dir, exist_ok=True)

    policy_net = ConvDQN().to(device)
    target_net = ConvDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult)
    policy = EpsilonGreedy(policy_net, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
    memory = ReplayMemory(memory_size)

    recent_scores = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    steps_done = 0
    start_episode = 0
    best_avg_score = 0

    model_path = os.path.join(save_dir, "model.pth")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        policy_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded model from {model_path}")

    game = Game()

    for episode in range(start_episode, num_episodes):
        board = game.reset()
        state = encode_board(board, device)

        while True:
            legal_actions = [a.value for a in game.get_legal_actions()]
            if not legal_actions:
                break

            action_idx, _ = policy(state, legal_actions, steps_done)
            action = legal_actions[action_idx]
            steps_done += 1

            score_gained, done = game.step(Action(action))

            if score_gained > 0:
                reward = score_gained / 100.0
            else:
                reward = 0.0

            if not done:
                next_board = game.board.copy()
                next_state = encode_board(next_board, device)
                next_legal_actions = [a.value for a in game.get_legal_actions()]
            else:
                next_state = None
                next_legal_actions = None

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

        recent_scores.append(game.score)
        recent_max_tiles.append(game.get_max_tile())

        loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size, gamma)
        scheduler.step()

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0

        if episode % save_interval == 0 and episode > 0:
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                torch.save(policy_net.state_dict(), os.path.join(save_dir, "best_model.pth"))

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

        if verbose and episode % 100 == 0:
            avg_max = sum(recent_max_tiles) / len(recent_max_tiles) if recent_max_tiles else 0
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            current_lr = scheduler.get_last_lr()[0]
            eps = policy.eps_end + (policy.eps_start - policy.eps_end) * \
                  math.exp(-steps_done / policy.eps_decay)
            print(
                f"Episode {episode:5d} | "
                f"Avg Score: {avg_score:8.1f} | "
                f"Avg Max Tile: {avg_max:6.1f} | "
                f"Loss: {loss_str} | "
                f"LR: {current_lr:.2e} | "
                f"Eps: {eps:.3f}"
            )

    torch.save(policy_net.state_dict(), model_path)
    if verbose:
        print(f"Training complete. Model saved to {model_path}")


def train_ntuple(
    num_episodes: int,
    save_dir: str,
    verbose: bool = True,
    save_interval: int = 1000,
    lr: float = NTUPLE_LR,
    lr_decay: float = NTUPLE_LR_DECAY,
) -> None:
    """Train N-tuple network with TD(0) learning.

    This implements the state-of-the-art approach for 2048:
    - Uses afterstate learning (state after move, before random tile)
    - TD(0) online updates at each step
    - 8-fold symmetry exploitation for better generalization

    Args:
        num_episodes: Number of training episodes
        save_dir: Directory to save models
        verbose: Whether to print progress
        save_interval: Save model every N episodes
        lr: Initial learning rate
        lr_decay: Learning rate decay per step
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize N-tuple network
    network = NTupleNetwork(learning_rate=lr)

    # Statistics
    recent_scores = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    tile_counts = {2048: 0, 4096: 0, 8192: 0, 16384: 0, 32768: 0}
    total_games = 0
    start_episode = 0
    best_avg_score = 0
    current_lr = lr

    # Paths
    model_path = os.path.join(save_dir, "ntuple_model.pkl")
    checkpoint_path = os.path.join(save_dir, "ntuple_checkpoint.pkl")

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        network.load(model_path)
        start_episode = checkpoint["episode"]
        total_games = checkpoint["total_games"]
        best_avg_score = checkpoint.get("best_avg_score", 0)
        current_lr = checkpoint.get("learning_rate", lr)
        tile_counts = checkpoint.get("tile_counts", tile_counts)
        network.learning_rate = current_lr
        if verbose:
            print(f"Resumed from episode {start_episode}")
            print(f"Current LR: {current_lr:.6f}")
            print(f"Tile achievements: {tile_counts}")
    elif os.path.exists(model_path):
        network.load(model_path)
        if verbose:
            print(f"Loaded model from {model_path}")

    if verbose:
        mem_usage = network.get_memory_usage_mb()
        print(f"N-tuple network memory usage: {mem_usage:.1f} MB")

    game = Game()

    for episode in range(start_episode, num_episodes):
        # Reset game
        game.reset()
        total_games += 1

        prev_afterstate = None
        prev_reward = 0

        while not game.is_game_over():
            legal_actions = game.get_legal_actions()
            if not legal_actions:
                break

            # Select best action using afterstate evaluation
            best_action = None
            best_value = float('-inf')
            best_afterstate = None
            best_reward = 0

            for action in legal_actions:
                afterstate, reward, moved = game.get_afterstate(action)
                if moved and afterstate is not None:
                    value = reward + network.evaluate(afterstate)
                    if value > best_value:
                        best_value = value
                        best_action = action
                        best_afterstate = afterstate
                        best_reward = reward

            if best_action is None:
                break

            # TD(0) update for previous afterstate
            # V(s_prev) <- V(s_prev) + alpha * (r_prev + V(s_curr) - V(s_prev))
            if prev_afterstate is not None:
                current_value = network.evaluate(best_afterstate)
                prev_value = network.evaluate(prev_afterstate)
                td_error = prev_reward + current_value - prev_value
                network.update(prev_afterstate, td_error)

            # Store current afterstate for next update
            prev_afterstate = best_afterstate.copy()
            prev_reward = best_reward

            # Execute action: first move, then spawn tile
            game.step_without_spawn(best_action)
            game.spawn_tile()

            # Decay learning rate
            current_lr *= lr_decay
            network.learning_rate = current_lr

        # Final update for terminal state (V(terminal) = 0)
        if prev_afterstate is not None:
            prev_value = network.evaluate(prev_afterstate)
            td_error = prev_reward + 0 - prev_value  # terminal value = 0
            network.update(prev_afterstate, td_error)

        # Record statistics
        score = game.score
        max_tile = game.get_max_tile()
        recent_scores.append(score)
        recent_max_tiles.append(max_tile)

        # Track tile achievements
        for tile in [32768, 16384, 8192, 4096, 2048]:
            if max_tile >= tile:
                tile_counts[tile] = tile_counts.get(tile, 0) + 1
                break

        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0

        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                network.save(os.path.join(save_dir, "ntuple_best_model.pkl"))

            network.save(model_path)
            import pickle
            checkpoint = {
                "episode": episode,
                "total_games": total_games,
                "best_avg_score": best_avg_score,
                "learning_rate": current_lr,
                "tile_counts": tile_counts,
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)

            if verbose:
                print(f"Checkpoint saved (episode {episode}, best avg: {best_avg_score:.1f})")

        # Print progress
        if verbose and episode % 100 == 0:
            avg_max = sum(recent_max_tiles) / len(recent_max_tiles) if recent_max_tiles else 0

            # Calculate tile achievement rates
            rate_2048 = tile_counts[2048] / total_games * 100 if total_games > 0 else 0
            rate_4096 = tile_counts[4096] / total_games * 100 if total_games > 0 else 0
            rate_8192 = tile_counts[8192] / total_games * 100 if total_games > 0 else 0
            rate_16384 = tile_counts[16384] / total_games * 100 if total_games > 0 else 0
            rate_32768 = tile_counts[32768] / total_games * 100 if total_games > 0 else 0

            print(
                f"Episode {episode:6d} | "
                f"Avg Score: {avg_score:8.1f} | "
                f"Avg Max: {avg_max:6.0f} | "
                f"LR: {current_lr:.2e} | "
                f"2048: {rate_2048:5.1f}% | "
                f"4096: {rate_4096:5.1f}% | "
                f"8192: {rate_8192:5.1f}%"
            )

            if rate_16384 > 0 or rate_32768 > 0:
                print(
                    f"         High tiles: "
                    f"16384: {rate_16384:.2f}% ({tile_counts[16384]}) | "
                    f"32768: {rate_32768:.2f}% ({tile_counts[32768]})"
                )

    # Save final model
    network.save(model_path)
    if verbose:
        print(f"Training complete. Model saved to {model_path}")
        print(f"Final tile achievements over {total_games} games:")
        for tile, count in sorted(tile_counts.items()):
            rate = count / total_games * 100 if total_games > 0 else 0
            print(f"  {tile}: {count} ({rate:.2f}%)")


# Keep the old train function for backward compatibility
def train(
    num_episodes: int,
    save_dir: str,
    device: torch.device,
    verbose: bool = True,
    save_interval: int = 1000,
    batch_size: int = BATCH_SIZE,
    gamma: float = GAMMA,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    target_update: int = TARGET_UPDATE,
    memory_size: int = MEMORY_SIZE,
    t_0: int = T_0,
    t_mult: int = T_MULT,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: int = 500000,
) -> None:
    """Train the DQN agent (backward compatible wrapper)."""
    train_dqn(
        num_episodes=num_episodes,
        save_dir=save_dir,
        device=device,
        verbose=verbose,
        save_interval=save_interval,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        weight_decay=weight_decay,
        target_update=target_update,
        memory_size=memory_size,
        t_0=t_0,
        t_mult=t_mult,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
    )


def main():
    parser = argparse.ArgumentParser(description="Train 2048 agent")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "ntuple"],
        default="ntuple",
        help="Model type: cnn (DQN) or ntuple (TD learning) (default: ntuple)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100000,
        help="Number of training episodes (default: 100000)",
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
        help="Device to use for DQN: cpu, cuda, mps, or auto (default: auto)",
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
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 0.1 for ntuple, 3e-4 for cnn)",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=NTUPLE_LR_DECAY,
        help=f"Learning rate decay per step for ntuple (default: {NTUPLE_LR_DECAY})",
    )
    args = parser.parse_args()

    if args.model == "ntuple":
        lr = args.lr if args.lr is not None else NTUPLE_LR

        if not args.quiet:
            print("=" * 60)
            print("N-tuple + TD Learning for 2048")
            print("=" * 60)
            print(f"Episodes: {args.episodes}")
            print(f"Learning rate: {lr}")
            print(f"LR decay: {args.lr_decay}")
            print(f"Save directory: {args.save_dir}")
            print("=" * 60)

        train_ntuple(
            num_episodes=args.episodes,
            save_dir=args.save_dir,
            verbose=not args.quiet,
            save_interval=args.save_interval,
            lr=lr,
            lr_decay=args.lr_decay,
        )
    else:
        # Select device for DQN
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        lr = args.lr if args.lr is not None else LR

        if not args.quiet:
            print(f"Using device: {device}")

        train_dqn(
            num_episodes=args.episodes,
            save_dir=args.save_dir,
            device=device,
            verbose=not args.quiet,
            save_interval=args.save_interval,
            lr=lr,
        )


if __name__ == "__main__":
    main()
