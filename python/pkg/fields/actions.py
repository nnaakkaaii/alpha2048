from enum import IntEnum


class Action(IntEnum):
    """2048 game actions: slide tiles in 4 directions."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


NUM_ACTIONS = len(Action)
