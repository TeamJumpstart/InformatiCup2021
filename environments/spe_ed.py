from dataclasses import dataclass
import numpy as np

actions = ("turn_left", "turn_right", "slow_down", "speed_up", "change_nothing")
directions = ("right", "up", "left", "down")
direction_angle = {
    "right": 0,
    "up": np.pi / 2,
    "left": np.pi,
    "down": np.pi * 3 / 2,
}
cartesian_directions = {
    "right": np.array([1, 0]),
    "up": np.array([0, 1]),
    "left": np.array([-1, 0]),
    "down": np.array([0, -1]),
}


def turn_left(direction):
    """Rotate a direction one turn to the left."""
    return directions[(directions.index(direction) + 1) % 4]


def turn_right(direction):
    """Rotate a direction one turn to the right."""
    return directions[(directions.index(direction) + 3) % 4]


@dataclass
class Player:
    """Player object."""
    player_id: int
    x: int
    y: int
    direction: str
    speed: int
    active: bool
    name: str = None

    @property
    def position(self):
        """Return current position as ndarray."""
        return np.array([self.x, self.y])


class Map:
    """Cell state wrapper for common methods."""
    def __init__(self, cells):
        self.cells = cells
        self.cells.setflags(write=False)  # Prevent accidentally writing
        self.width = cells.shape[0]
        self.height = cells.shape[1]

    def __getitem__(self, key):
        x, y = key
        # Check if index is inside bounds
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[x, y]
        return None  # Otherwise return None

    def is_free(self, pos):
        """Check if target location is not occupied"""
        return self[pos[0], pos[1]] == 0
