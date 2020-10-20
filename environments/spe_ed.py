from dataclasses import dataclass
import typing
import json
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
        return directions[(directions.index(self) + 3) % 4]

    def turn_right(self):
        """Rotates one turn to the right."""
        return directions[(directions.index(self) + 1) % 4]

    def __repr__(self):
        return self.name


directions = (
    Direction("right", 0, np.array([1, 0])),
    Direction("down", np.pi / 2, np.array([0, 1])),
    Direction("left", np.pi, np.array([-1, 0])),
    Direction("up", np.pi * 3 / 2, np.array([0, -1])),
)
directions_by_name = {d.name: d for d in directions}


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

    def copy(self):
        """Create a copy of this player."""
        return Player(self.player_id, self.x, self.y, self.direction, self.speed, self.active, self.name)

    def perform(self, action):
        """Let the player perform one action.

        This does not actually move the player.
        """
        if not self.active:
            return

        if action == 'turn_left':
            self.direction = self.direction.turn_left()
        elif action == 'turn_right':
            self.direction = self.direction.turn_right()
        elif action == 'slow_down':
            if self.speed <= 1:  # Check minimum speed
                self.active = False
            else:
                self.speed -= 1
        elif action == 'speed_up':
            if self.speed >= 10:  # Check maximum speed
                self.active = False
            else:
                self.speed += 1

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (
                self.player_id == other.player_id and self.x == other.x and self.y == other.y and
                self.direction == other.direction and self.speed == other.speed and self.active == other.active
            )
        return False

    @classmethod
    def from_json(cls, player_id, data):
        """Create Player object from json data."""
        return cls(
            player_id=int(player_id),
            x=data['x'],
            y=data['y'],
            direction=directions_by_name[data['direction']],
            speed=data['speed'],
            active=data['active'],
            name=data.get('name'),
        )


class Map:
    """Cell state wrapper for common methods."""
    def __init__(self, cells):
        self.cells = cells
        self.cells.setflags(write=False)  # Prevent accidentally writing
        self.width = cells.shape[0]
        self.height = cells.shape[1]

    def __getitem__(self, key):
        y, x = key
        # Check if index is inside bounds
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[y, x]
        return None  # Otherwise return None

    def is_free(self, pos):
        """Check if target location is not occupied"""
        return self[pos[1], pos[0]] == 0


def infer_action(player_before, player_after):
    """Reconstruct that action that leads to the next state."""
    if player_after.speed > player_before.speed:
        return "speed_up"
    if player_after.speed < player_before.speed:
        return "slow_down"
    if player_after.direction == player_before.direction.turn_left():
        return "turn_left"
    if player_after.direction == player_before.direction.turn_right():
        return "turn_right"
    return "change_nothing"


class SavedGame:
    """Save game representation

    Complete game, that sonsist of a series of states.
    Each state contains:
    * width/height: Redundant with cells.shape
    * cells
    * players
    * you: Irrelevant
    * running: Redundant, last state has running=False
    * deadline: irrelevant
    """
    def __init__(self, data):
        self.data = data
        self.cell_states = [np.array(state['cells'], dtype=np.int8) for state in data]
        self.player_states = [
            [Player.from_json(player_id, player_data) for player_id, player_data in state['players'].items()]
            for state in data
        ]
        self.height, self.width = self.cell_states[0].shape

    def infer_actions(self, t):
        """Compute action taken by all players at timestep `t`."""
        actions = []
        for i in range(len(self.player_states[0])):
            actions.append(infer_action(self.player_states[t][i], self.player_states[t + 1][i]))
        return actions

    @property
    def winner(self):
        """Winner is the last surviving player."""
        active = [p for p in self.player_states[-1] if p.active]
        assert len(active) == 1
        return active[0]

    @property
    def rounds(self):
        """Number of total rounds played."""
        return len(self.cell_states) - 1

    @classmethod
    def load(cls, file_name):
        """"Load a saved game.

        Args:
            file_name: Path to the save game in json format.

        Returns:
            SavedGame object
        """
        with open(file_name) as f:
            data = json.load(f)

        if data[-1]["running"]:
            raise ValueError(f"Game not completed: {file_name}")

        return cls(data)
