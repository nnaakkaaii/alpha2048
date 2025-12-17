#!/usr/bin/env python3
"""Test trained 2048 DQN agent."""
import argparse
import os
import time

import torch

from pkg.fields import Action, Game
from pkg.networks import MLP
from pkg.policies import Greedy
from pkg.utils import encode_board


def test(
    model_path: str,
    device: torch.device,
    num_games: int = 10,
    verbose: bool = True,
    delay: float = 0.0,
) -> dict:
    """Test the trained agent."""
    # Load model
    policy_net = MLP().to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        policy_net.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random weights")

    policy_net.eval()
    policy = Greedy(policy_net)

    game = Game()
    results = []

    for game_num in range(num_games):
        board = game.reset()
        state = encode_board(board, device)
        steps = 0

        if verbose:
            print(f"\n=== Game {game_num + 1}/{num_games} ===")

        while True:
            legal_actions = [a.value for a in game.get_legal_actions()]
            if not legal_actions:
                break

            action_idx, score = policy(state, legal_actions, 0)
            action = legal_actions[action_idx]

            _, done = game.step(Action(action))
            steps += 1

            if verbose and delay > 0:
                print("\033[2J\033[H", end="")
                print(f"Game {game_num + 1}/{num_games} | Step {steps}")
                print(f"Action: {Action(action).name} | Q-value: {score:.2f}")
                print(game.render())
                time.sleep(delay)

            if done:
                break

            board = game.board.copy()
            state = encode_board(board, device)

        result = {
            "score": game.score,
            "max_tile": game.get_max_tile(),
            "steps": steps,
        }
        results.append(result)

        if verbose:
            print(f"Score: {game.score} | Max Tile: {game.get_max_tile()} | Steps: {steps}")

    # Compute statistics
    scores = [r["score"] for r in results]
    max_tiles = [r["max_tile"] for r in results]
    steps_list = [r["steps"] for r in results]

    stats = {
        "num_games": num_games,
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "avg_max_tile": sum(max_tiles) / len(max_tiles),
        "best_max_tile": max(max_tiles),
        "avg_steps": sum(steps_list) / len(steps_list),
        "tile_2048_count": sum(1 for t in max_tiles if t >= 2048),
        "tile_1024_count": sum(1 for t in max_tiles if t >= 1024),
        "tile_512_count": sum(1 for t in max_tiles if t >= 512),
    }

    if verbose:
        print("\n=== Summary ===")
        print(f"Games: {stats['num_games']}")
        print(f"Avg Score: {stats['avg_score']:.1f}")
        print(f"Max Score: {stats['max_score']}")
        print(f"Min Score: {stats['min_score']}")
        print(f"Avg Max Tile: {stats['avg_max_tile']:.1f}")
        print(f"Best Max Tile: {stats['best_max_tile']}")
        print(f"Avg Steps: {stats['avg_steps']:.1f}")
        print(f"Reached 2048: {stats['tile_2048_count']}/{num_games}")
        print(f"Reached 1024: {stats['tile_1024_count']}/{num_games}")
        print(f"Reached 512: {stats['tile_512_count']}/{num_games}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Test trained 2048 DQN agent")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model.pth",
        help="Path to model file (default: checkpoints/model.pth)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of test games (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between moves in seconds for visualization (default: 0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    test(
        model_path=args.model,
        device=device,
        num_games=args.games,
        verbose=not args.quiet,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
