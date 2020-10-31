import numpy as np
import gym
from environments.spe_ed import Map


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
        self.rounds = 1

        self.viewer = None

    def render(self, mode='human', screen_width=720, screen_height=720):
        from visualization import Spe_edAx
        import matplotlib.pyplot as plt

        if self.viewer is None:
            fig = plt.figure(
                figsize=(screen_width / 100, screen_height / 100),
                dpi=100,
                tight_layout=True,
            )
            ax = plt.subplot(1, 1, 1)
            self.viewer = Spe_edAx(fig, ax, self.cells, self.players)

            if mode == 'human':
                plt.show(block=False)
        else:
            self.viewer.update(self.cells, self.players)

        if mode == 'human':
            plt.pause(1e-6)  # Let plot handlers resolve and update window
            return not self.viewer.closed
        elif mode == 'rgb_array':
            # Redraw and fetch rgb-array from plot
            fig = plt.gcf()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

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
        return Map(occupancy), you, opponents, self.rounds
