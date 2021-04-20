from dataclasses import dataclass
import json
import time
import numpy as np

actions = ("turn_left", "turn_right", "slow_down", "speed_up", "change_nothing")


@dataclass(frozen=True)
class Direction(np.lib.mixins.NDArrayOperatorsMixin):
    """Common operations for directions."""
    index: int
    name: str
    angle: float
    cartesian: np.ndarray

    def turn_left(self):
        """Rotates one turn to the left."""
        return directions[(self.index + 3) % 4]

    def turn_right(self):
        """Rotates one turn to the right."""
        return directions[(self.index + 1) % 4]

    def __repr__(self):
        return self.name

    def __array__(self):
        """Return numpy compatible representation."""
        return self.cartesian


directions = np.empty(4, dtype=object)
directions[0] = Direction(0, "right", 0, np.array([1, 0]))
directions[1] = Direction(1, "down", np.pi / 2, np.array([0, 1]))
directions[2] = Direction(2, "left", np.pi, np.array([-1, 0]))
directions[3] = Direction(3, "up", np.pi * 3 / 2, np.array([0, -1]))
directions.setflags(write=False)  # Prevent accidentally writing
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
            self.speed -= 1
            if self.speed < 1:  # Check minimum speed
                self.active = False
        elif action == 'speed_up':
            self.speed += 1
            if self.speed > 10:  # Check maximum speed
                self.active = False
        elif action == 'change_nothing':
            pass
        else:  # Invalid
            self.active = False

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (
                self.player_id == other.player_id and self.x == other.x and self.y == other.y and
                self.direction == other.direction and self.speed == other.speed and self.active == other.active
            )
        return False

    def __str__(self):
        return f"{self.player_id}: ({self.x}, {self.y}), {self.direction}, speed={self.speed}, active={self.active}"

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

    def to_dict(self):
        """Serialize Player object for JSON storage."""
        d = {
            'x': int(self.x),  # Cast to python int
            'y': int(self.y),
            'direction': self.direction.name,
            'speed': self.speed,
            'active': self.active,
        }
        if self.name is not None:  # Add name if present
            d['name'] = self.name

        return str(self.player_id), d


class Cells(np.ndarray):
    """Cell state wrapper for common methods."""
    def __new__(cls, cells):
        return cells.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def is_free(self, position):
        """Check if target location is not occupied."""
        x, y = position
        # Check if index is inside bounds
        if 0 <= x < self.width and 0 <= y < self.height:
            return self[y, x] == 0
        return False


def infer_action(player_before, player_after):
    """Reconstruct that action that leads to the next state."""
    if not player_before.active:
        return "inactive"
    if player_after.speed > player_before.speed:
        return "speed_up"
    if player_after.speed < player_before.speed:
        return "slow_down"
    if player_after.direction == player_before.direction.turn_left():
        return "turn_left"
    if player_after.direction == player_before.direction.turn_right():
        return "turn_right"
    if player_before.x == player_after.x and player_before.y == player_after.y and not player_after.active:
        return "invalid"
    return "change_nothing"


class SavedGame:
    """Save game representation.

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
        """Initialize SavedGame.

        Args:
            data: JSON object
        """
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
        """Winner is the last surviving player.

        Returns `None` on draw.
        """
        active = [p for p in self.player_states[-1] if p.active]
        return active[0] if len(active) > 0 else None

    @property
    def rounds(self):
        """Number of total rounds played."""
        return len(self.cell_states) - 1

    @property
    def names(self):
        """Return list of all player names in this game."""
        return [p.name for p in self.player_states[-1]]

    @property
    def player_ids(self):
        """Return iterable of all player ids in this game."""
        return (p.player_id for p in self.player_states[0])

    @property
    def you(self):
        """Player_id of controlled player."""
        return self.data[0]["you"]

    @classmethod
    def load(cls, file_name):
        """Load a saved game.

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

    def create_simulator(self, t):
        """Initialize a simulate with the gamestate at time `t`."""
        from environments.simulator import Spe_edSimulator

        return Spe_edSimulator(self.cell_states[t], self.player_states[t], t + 1)

    def move_controlled_player_to_front(self):
        """Changes cell and player states, as if controlled player was at first position."""
        you = self.you
        if you is None:
            raise ValueError("The is no controlled player in this game.")
        if you != 1:
            for t in range(len(self.data)):
                # Swap cells
                cells = self.cell_states[t]
                your_cells = cells == you
                other_cells = cells == 1
                cells[your_cells] = 1
                cells[other_cells] = you

                # Swap players
                players = self.player_states[t]
                players[0], players[you - 1] = players[you - 1], players[0]

    def get_obs(self, t, player_id, time_limit=5):
        """Get obersation from the perspective of a specific player.

        Returned values can be used as input for a policy.

        Args:
            t: timestep, zero-indexed, so t=0 equals rounds=1
            player_id: Id of the player to get the observation for
        """
        occupancy = self.cell_states[t] != 0
        occupancy.setflags(write=False)  # Prevent accidentally writing
        you = self.player_states[t][player_id - 1]
        opponents = [p for p in self.player_states[t] if p.active and p.player_id != player_id]
        deadline = time.time() + time_limit
        return Cells(occupancy), you, opponents, t + 1, deadline
