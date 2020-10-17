from dataclasses import dataclass
import numpy as np

actions = ("turn_left", "turn_right", "slow_down", "speed_up", "change_nothing")


@dataclass(frozen=True)
class Direction:
    """Common operations for directions"""
    name: str
    angle: float
    cartesian: np.ndarray

    def turn_left(self):
        """Rotates one turn to the left."""
        return directions[(directions.index(self) + 1) % 4]

    def turn_right(self):
        """Rotates one turn to the right."""
        return directions[(directions.index(self) + 3) % 4]


directions = (
    Direction("right", 0, np.array([1, 0])),
    Direction("up", np.pi / 2, np.array([0, 1])),
    Direction("left", np.pi, np.array([-1, 0])),
    Direction("down", np.pi * 3 / 2, np.array([0, -1])),
)


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
