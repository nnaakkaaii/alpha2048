#!/usr/bin/env python3
"""Automatic 2048 play with random policy."""
import argparse
import random
import time

from pkg.fields import Action, Game


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def random_policy(legal_actions: list[Action]) -> Action:
    """Select random action from legal actions."""
    return random.choice(legal_actions)


def main():
    """Run automatic 2048 game."""
    parser = argparse.ArgumentParser(description="Auto-play 2048 with random policy")
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="Delay between moves in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step output",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    delay_sec = args.delay / 1000.0
    action_names = {
        Action.UP: "UP",
        Action.DOWN: "DOWN",
        Action.LEFT: "LEFT",
        Action.RIGHT: "RIGHT",
    }

    results = []

    for game_num in range(args.games):
        game = Game()
        game.reset()
        steps = 0

        if not args.quiet:
            if args.games > 1:
                print(f"\n=== Game {game_num + 1}/{args.games} ===")
            else:
                print("=== 2048 Auto-Play ===")
            print()

        while not game.is_game_over():
            legal_actions = game.get_legal_actions()
            if not legal_actions:
                break

            action = random_policy(legal_actions)
            game.step(action)
            steps += 1

            if not args.quiet:
                clear_screen()
                print(f"Step: {steps}, Action: {action_names[action]}")
                print(game.render())
                time.sleep(delay_sec)

        results.append({
            "score": game.score,
            "max_tile": game.get_max_tile(),
            "steps": steps,
        })

        if not args.quiet:
            print("Game Over!")
            print(f"Final Score: {game.score}")
            print(f"Max Tile: {game.get_max_tile()}")
            print(f"Total Steps: {steps}")

    # Print summary for multiple games
    if args.games > 1:
        print("\n=== Summary ===")
        scores = [r["score"] for r in results]
        max_tiles = [r["max_tile"] for r in results]
        steps_list = [r["steps"] for r in results]

        print(f"Games: {args.games}")
        print(f"Avg Score: {sum(scores) / len(scores):.1f}")
        print(f"Max Score: {max(scores)}")
        print(f"Min Score: {min(scores)}")
        print(f"Avg Max Tile: {sum(max_tiles) / len(max_tiles):.1f}")
        print(f"Best Max Tile: {max(max_tiles)}")
        print(f"Avg Steps: {sum(steps_list) / len(steps_list):.1f}")


if __name__ == "__main__":
    main()
