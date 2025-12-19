#!/usr/bin/env python3
"""Enhanced training script with full hyperparameter control."""
import argparse
import os
from collections import deque
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    OneCycleLR,
    ReduceLROnPlateau
)

from pkg.fields import Action, Game
from pkg.networks import ConvDQN
from pkg.policies import EpsilonGreedy
from pkg.utils import ReplayMemory, Transition, encode_board


def optimize_model(
    memory: ReplayMemory,
    policy_net: ConvDQN,
    target_net: ConvDQN,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int = 512,
    gamma: float = 0.99,
    gradient_clip: float = 1.0,
    double_dqn: bool = True,
    accumulation_steps: int = 1,
    step_counter: int = 0,
) -> Optional[float]:
    """Perform one optimization step with configurable parameters."""
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
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
    next_state_values = torch.zeros(batch_size, device=device)

    if non_final_mask.sum() > 0:
        with torch.no_grad():
            if double_dqn:
                # Double DQN: use policy_net to select action, target_net to evaluate
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
            else:
                # Standard DQN
                target_q = target_net(non_final_next_states)
                next_state_values[non_final_mask] = target_q.max(1)[0]

    # Compute expected Q values
    expected_state_action_values = reward_batch + (gamma * next_state_values)

    # Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Gradient accumulation
    if accumulation_steps > 1:
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step_counter + 1) % accumulation_steps == 0:
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), gradient_clip)
            optimizer.step()
            optimizer.zero_grad()
    else:
        # Standard optimization
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), gradient_clip)
        optimizer.step()

    return loss.item() * accumulation_steps  # Return unscaled loss


def train(
    num_episodes: int,
    save_dir: str = "checkpoints",
    device: Optional[torch.device] = None,
    verbose: bool = True,
    save_interval: int = 1000,
    # Hyperparameters
    batch_size: int = 512,
    gamma: float = 0.99,
    target_update: int = 100,
    memory_size: int = 1000000,
    # Optimizer settings
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    optimizer_type: str = "adamw",
    momentum: float = 0.9,  # For SGD
    betas: tuple = (0.9, 0.999),  # For Adam/AdamW
    # Scheduler settings
    scheduler_type: str = "cosine",
    t_0: int = 1000,  # For cosine annealing
    t_mult: int = 2,  # For cosine annealing
    step_size: int = 1000,  # For step scheduler
    scheduler_gamma: float = 0.9,  # For step/exponential scheduler
    patience: int = 100,  # For ReduceLROnPlateau
    # Epsilon greedy settings
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: int = 500000,
    # Training settings
    gradient_clip: float = 1.0,
    double_dqn: bool = True,
    # Multi-episode batch training
    episodes_per_update: int = 1,
    updates_per_episode: int = 1,
    accumulation_steps: int = 1,
    # Reward shaping
    reward_scale: float = 100.0,
    # Early stopping
    early_stop_patience: int = 0,  # 0 = disabled
    early_stop_min_delta: float = 0.0,
) -> Dict[str, Any]:
    """Train the DQN agent with full hyperparameter control.
    
    Args:
        num_episodes: Total number of episodes to train
        save_dir: Directory to save model checkpoints
        device: Device to train on (cuda/cpu/mps), None for auto
        verbose: Whether to print training progress
        save_interval: Save model every N episodes
        batch_size: Batch size for training (higher = more GPU memory)
        gamma: Discount factor for rewards
        target_update: Update target network every N episodes
        memory_size: Size of replay buffer
        lr: Learning rate
        weight_decay: L2 regularization weight
        optimizer_type: adam, adamw, sgd, rmsprop
        momentum: Momentum for SGD
        betas: Betas for Adam/AdamW
        scheduler_type: cosine, step, exponential, onecycle, plateau, none
        t_0: Period for cosine annealing
        t_mult: Period multiplier for cosine annealing
        step_size: Step size for step scheduler
        scheduler_gamma: Decay factor for step/exponential scheduler
        patience: Patience for ReduceLROnPlateau
        eps_start: Initial exploration rate
        eps_end: Final exploration rate
        eps_decay: Exploration decay rate
        gradient_clip: Gradient clipping value (0 = disabled)
        double_dqn: Use Double DQN
        episodes_per_update: Collect N episodes before updating
        updates_per_episode: Perform N updates per episode
        accumulation_steps: Gradient accumulation steps
        reward_scale: Scale factor for rewards
        early_stop_patience: Stop if no improvement for N episodes (0 = disabled)
        early_stop_min_delta: Minimum improvement to reset patience
        
    Returns:
        Dictionary with training history and final model
    """
    # Auto-select device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    if verbose:
        print(f"Training configuration:")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Optimizer: {optimizer_type}")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Double DQN: {double_dqn}")
        print(f"  Gradient accumulation: {accumulation_steps}")
        print(f"  Episodes per update: {episodes_per_update}")
        print(f"  Updates per episode: {updates_per_episode}")
        print(f"  Memory size: {memory_size}")
        print()

    os.makedirs(save_dir, exist_ok=True)

    # Initialize networks
    policy_net = ConvDQN().to(device)
    target_net = ConvDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Initialize optimizer
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            policy_net.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            policy_net.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            policy_net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    elif optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            policy_net.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Initialize scheduler
    scheduler = None
    if scheduler_type.lower() == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult)
    elif scheduler_type.lower() == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=scheduler_gamma)
    elif scheduler_type.lower() == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif scheduler_type.lower() == "onecycle":
        total_steps = num_episodes * updates_per_episode
        scheduler = OneCycleLR(
            optimizer, max_lr=lr*10, total_steps=total_steps, pct_start=0.3
        )
    elif scheduler_type.lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', patience=patience, factor=scheduler_gamma
        )

    # Initialize policy and memory
    policy = EpsilonGreedy(policy_net, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay)
    memory = ReplayMemory(memory_size)

    # Statistics
    recent_scores = deque(maxlen=100)
    recent_max_tiles = deque(maxlen=100)
    history = {
        'scores': [],
        'max_tiles': [],
        'losses': [],
        'lrs': [],
        'epsilons': []
    }
    
    steps_done = 0
    start_episode = 0
    best_avg_score = 0
    update_counter = 0
    
    # Early stopping
    best_early_stop_score = 0
    early_stop_counter = 0

    # Load checkpoint if available
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy_net.load_state_dict(checkpoint["model_state"])
        target_net.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        steps_done = checkpoint["steps_done"]
        start_episode = checkpoint["episode"]
        best_avg_score = checkpoint.get("best_avg_score", 0)
        if verbose:
            print(f"Resumed from episode {start_episode}, steps {steps_done}")

    game = Game()
    episode_buffer = []

    for episode in range(start_episode, num_episodes):
        board = game.reset()
        state = encode_board(board, device)
        episode_transitions = []

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

            # Reward shaping
            if score_gained > 0:
                reward = score_gained / reward_scale
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
            transition = (
                state,
                torch.tensor([[action]], device=device, dtype=torch.long),
                next_state,
                next_legal_actions,
                torch.tensor([reward], device=device, dtype=torch.float32),
            )
            episode_transitions.append(transition)
            memory.push(*transition)

            state = next_state

            if done:
                break

        # Record statistics
        recent_scores.append(game.score)
        recent_max_tiles.append(game.get_max_tile())
        history['scores'].append(game.score)
        history['max_tiles'].append(game.get_max_tile())

        # Training updates
        if (episode + 1) % episodes_per_update == 0 and len(memory) >= batch_size:
            losses = []
            for update_idx in range(updates_per_episode):
                loss = optimize_model(
                    memory, policy_net, target_net, optimizer, device,
                    batch_size=batch_size,
                    gamma=gamma,
                    gradient_clip=gradient_clip,
                    double_dqn=double_dqn,
                    accumulation_steps=accumulation_steps,
                    step_counter=update_counter
                )
                update_counter += 1
                if loss is not None:
                    losses.append(loss)
                    history['losses'].append(loss)
            
            # Scheduler step
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                    scheduler.step(avg_score)
                elif not isinstance(scheduler, OneCycleLR):
                    scheduler.step()
            
            # OneCycle scheduler steps per update
            if isinstance(scheduler, OneCycleLR) and losses:
                for _ in range(len(losses)):
                    scheduler.step()

        # Update target network
        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Record learning rate and epsilon
        current_lr = optimizer.param_groups[0]['lr']
        current_eps = policy.eps_end + (policy.eps_start - policy.eps_end) * \
                     __import__("math").exp(-steps_done / policy.eps_decay)
        history['lrs'].append(current_lr)
        history['epsilons'].append(current_eps)

        # Calculate current stats
        avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
        
        # Early stopping check
        if early_stop_patience > 0:
            if avg_score > best_early_stop_score + early_stop_min_delta:
                best_early_stop_score = avg_score
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping triggered at episode {episode}")
                    break

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                torch.save(policy_net.state_dict(), os.path.join(save_dir, "dqn_best.pth"))

            checkpoint = {
                "model_state": policy_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "steps_done": steps_done,
                "episode": episode + 1,
                "best_avg_score": best_avg_score,
                "history": history,
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(policy_net.state_dict(), os.path.join(save_dir, "dqn_final.pth"))
            
            if verbose:
                print(f"Checkpoint saved (episode {episode + 1}, best avg: {best_avg_score:.1f})")

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_max = sum(recent_max_tiles) / len(recent_max_tiles) if recent_max_tiles else 0
            recent_losses = history['losses'][-100:] if history['losses'] else []
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            
            print(
                f"Episode {episode + 1:5d}/{num_episodes} | "
                f"Avg Score: {avg_score:8.1f} | "
                f"Avg Max Tile: {avg_max:6.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Eps: {current_eps:.3f}"
            )

    # Save final model
    torch.save(policy_net.state_dict(), os.path.join(save_dir, "dqn_final.pth"))
    if verbose:
        print(f"Training complete. Models saved to {save_dir}")

    return {
        "model": policy_net,
        "history": history,
        "final_avg_score": sum(recent_scores) / len(recent_scores) if recent_scores else 0,
        "best_avg_score": best_avg_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train 2048 DQN agent with enhanced hyperparameter control"
    )
    
    # Basic arguments
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save model every N episodes")
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target-update", type=int, default=100, help="Target network update frequency")
    parser.add_argument("--memory-size", type=int, default=1000000, help="Replay buffer size")
    
    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999], help="Betas for Adam")
    
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler type")
    parser.add_argument("--t0", type=int, default=1000, help="T_0 for cosine annealing")
    parser.add_argument("--t-mult", type=int, default=2, help="T_mult for cosine annealing")
    parser.add_argument("--step-size", type=int, default=1000, help="Step size for StepLR")
    parser.add_argument("--scheduler-gamma", type=float, default=0.9, help="Scheduler gamma")
    parser.add_argument("--patience", type=int, default=100, help="Patience for ReduceLROnPlateau")
    
    # Exploration
    parser.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--eps-end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--eps-decay", type=int, default=500000, help="Epsilon decay rate")
    
    # Training settings
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--no-double-dqn", action="store_true", help="Disable Double DQN")
    parser.add_argument("--episodes-per-update", type=int, default=1, help="Episodes before update")
    parser.add_argument("--updates-per-episode", type=int, default=1, help="Updates per episode")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--reward-scale", type=float, default=100.0, help="Reward scaling factor")
    
    # Early stopping
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Early stopping patience")
    parser.add_argument("--early-stop-delta", type=float, default=0.0, help="Min improvement delta")
    
    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        device = None
    else:
        device = torch.device(args.device)

    # Train with enhanced parameters
    results = train(
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        device=device,
        verbose=not args.quiet,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update=args.target_update,
        memory_size=args.memory_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        momentum=args.momentum,
        betas=tuple(args.betas),
        scheduler_type=args.scheduler,
        t_0=args.t0,
        t_mult=args.t_mult,
        step_size=args.step_size,
        scheduler_gamma=args.scheduler_gamma,
        patience=args.patience,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        gradient_clip=args.gradient_clip,
        double_dqn=not args.no_double_dqn,
        episodes_per_update=args.episodes_per_update,
        updates_per_episode=args.updates_per_episode,
        accumulation_steps=args.accumulation_steps,
        reward_scale=args.reward_scale,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_delta,
    )
    
    if not args.quiet:
        print(f"\nFinal average score: {results['final_avg_score']:.1f}")
        print(f"Best average score: {results['best_avg_score']:.1f}")


if __name__ == "__main__":
    main()