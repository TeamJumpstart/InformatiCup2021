from dataclasses import dataclass
import numpy as np
import gym

actions = ("turn_left", "turn_right", "slow_down", "speed_up", "change_nothing")
directions = ("right", "up", "left", "down")
direction_angle = {
    "right": 0,
    "up": np.pi / 2,
    "left": np.pi,
    "down": np.pi * 3 / 2,
}
cartesian_directions = {
    "right": (1, 0),
    "up": (0, 1),
    "left": (-1, 0),
    "down": (0, -1),
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


class Spe_edEnv(gym.Env):
    """Base class for Spe_ed environments.

    Handles common operations like rendering.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Copy of game state
        self.cells = np.empty((self.width, self.height), dtype=np.int8)
        self.players = []
        self.controlled_player = None
        self.round = 1

        self.viewer = None

    def render(self, mode='human', screen_width=720, screen_height=720):
        """Render the Spe_ed game state.

        Uses the binding provided by Gym for now, although it's not terribly efficient.
        """
        from gym.envs.classic_control import rendering

        # Create viewer if neccesary
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        # Draw occupied cells
        player_colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)
        ]
        collision_color = (0.0, 0.0, 0.0)
        xs = np.linspace(1, screen_width, self.cells.shape[0] + 1)  # Cell borders
        ys = np.linspace(1, screen_height, self.cells.shape[1] + 1)[::-1]
        for x in range(self.width):
            for y in range(self.height):
                cell_state = self.cells[x, y]
                if cell_state != 0:
                    color = player_colors[(cell_state - 1) % len(player_colors)] if cell_state > 0 else collision_color
                    self.viewer.draw_polygon(  # Quad for current cell
                        [(xs[x], ys[y + 1]), (xs[x + 1], ys[y + 1]), (xs[x + 1], ys[y]), (xs[x], ys[y])],
                        color=color,
                    )

        # Draw player directions
        arrow_lines = [  # Construct arrow
            rendering.Line((-0.4, 0.0), (0.4, 0.0)),
            rendering.Line((0.1, -0.3), (0.4, 0.0)),
            rendering.Line((0.4, 0.0), (0.1, 0.3)),
        ]
        for geom in arrow_lines:  # Set arrow linewidth
            geom.linewidth.stroke = 1.5

        for player in self.players:
            if not player.active:  # No direction for inactive players
                continue
            # Rotate and position arrow
            arrow = rendering.Compound(arrow_lines)
            arrow.add_attr(rendering.Transform(rotation=direction_angle[player.direction]))
            arrow.add_attr(rendering.Transform(translation=(0.5, 0.5)))
            arrow.add_attr(
                rendering.Transform(
                    translation=(xs[player.x], ys[player.y]),
                    scale=(xs[1] - xs[0], ys[1] - ys[0]),
                )
            )
            self.viewer.add_onetime(arrow)

        # Draw grid
        color_grid = (0.5, 0.5, 0.5)
        for x in xs:
            self.viewer.draw_line(start=(x, ys[0]), end=(x, ys[-1]), color=color_grid)
        for y in ys:
            self.viewer.draw_line(start=(xs[0], y), end=(xs[-1], y), color=color_grid)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _validate_action(self, action):
        """Change illegal actions to do nothing"""
        controlled_player = self.players[0]
        if controlled_player.speed >= 10 and action == "speed_up":
            action = "change_nothing"
        elif controlled_player.speed <= 1 and action == "slow_down":
            action = "change_nothing"
        return action

    def _get_obs(self, player):
        """Get obersation from the perspective of a specific player.

        Returned values can be used as input for a policy.

        Args:
            player_id: Id of the player to get the observation for
        """
        occupancy = self.cells != 0
        you = player
        opponents = [p for p in self.players if p.active and p.player_id != player.player_id]
        return occupancy, you, opponents, self.round
