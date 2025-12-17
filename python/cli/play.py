#!/usr/bin/env python3
"""Interactive 2048 game - play with keyboard (w/a/s/d)."""
import sys

from pkg.fields import Action, Game


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def parse_direction(key: str) -> Action | None:
    """Parse keyboard input to action."""
    key = key.lower().strip()
    mapping = {
        "w": Action.UP,
        "s": Action.DOWN,
        "a": Action.LEFT,
        "d": Action.RIGHT,
    }
    return mapping.get(key)


def main():
    """Run interactive 2048 game."""
    game = Game()
    game.reset()

    print("=== 2048 ===")
    print("Controls: w=Up, s=Down, a=Left, d=Right, q=Quit")
    print()

    while True:
        print(game.render())

        if game.is_game_over():
            print("Game Over!")
            print(f"Final Score: {game.score}")
            print(f"Max Tile: {game.get_max_tile()}")
            break

        try:
            user_input = input("Move: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nQuit.")
            break

        if user_input == "q":
            print("Quit.")
            break

        action = parse_direction(user_input)
        if action is None:
            print("Invalid input. Use w/a/s/d or q to quit.")
            continue

        # Check if move is legal
        legal_actions = game.get_legal_actions()
        if action not in legal_actions:
            print("Cannot move in that direction.")
            continue

        game.step(action)
        print()


if __name__ == "__main__":
    main()
