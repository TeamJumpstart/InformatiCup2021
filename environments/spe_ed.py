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
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (255, 1.0, 0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)
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
