import numpy as np
from environments.spe_ed import Player, directions
from environments.spe_ed_env import Spe_edEnv


def simulate(cells, players, rounds, actions):
    """Perfroma one game step of Spe_ed."""
    height, width = cells.shape

    # Perform actions
    for player, action in zip(players, actions):
        player.perform(action)

    # Move players
    newly_occupied = {}
    for player in players:
        if not player.active:
            continue
        pos = player.position
        for i in range(player.speed):
            pos += player.direction.cartesian
            if pos[0] < 0 or pos[1] < 0 or pos[1] >= height or pos[0] >= width:
                # Player left bounds
                player.active = False
                break  # Position after leaving the bouds is not actual position, just the first outside the bounds

            # Check for jumps
            if rounds % 6 == 0 and i > 0 and i < player.speed - 1:
                continue

            if cells[pos[1], pos[0]] != 0:
                # Collision
                player.active = False
                cells[pos[1], pos[0]] = -1

                if tuple(pos) in newly_occupied:  # Occupancy is from this round
                    newly_occupied[tuple(pos)].active = False  # Other player loses, too
            else:
                # No collision
                cells[pos[1], pos[0]] = player.player_id
                newly_occupied[tuple(pos)] = player  # Remember this cell
        player.x = pos[0]
        player.y = pos[1]

    # Round completed
    rounds += 1

    return cells, players, rounds, newly_occupied.keys()


class SimulatedSpe_edEnv(Spe_edEnv):
    def __init__(self, width, height, opponent_policies, seed=None):
        Spe_edEnv.__init__(self, width, height)

        self.opponent_policies = opponent_policies
        self.seed(seed)

    def step(self, action):
        # Gather actions for all players (do this before game state changes)
        actions = [action]
        for player in self.players[1:]:  # Compute actions of opponents
            if player.active:
                policy = self.opponent_policies[player.player_id - 2]
                obs = self._get_obs(player)
                actions.append(policy.act(*obs))
            else:
                actions.append("change_nothing")

        # Perform simulation step
        _, _, self.rounds, _ = simulate(self.cells, self.players, self.rounds, actions)

        done = sum(1 for p in self.players if p.active) < 2
        reward = 1 if done and self.controlled_player.active else 0
        return self._get_obs(self.controlled_player), reward, done, {}

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Generate a new game"""
        self.rounds = 1
        self.cells[:] = 0  # Clear occupancies
        self.players.clear()

        # Generate players
        for i in range(len(self.opponent_policies) + 1):
            player_id = i + 1

            # Choose startion position
            x = self.rng.integers(0, self.width)
            y = self.rng.integers(0, self.height)
            while self.cells[y, x] != 0:  # Ensure chosen location is empty
                x = self.rng.integers(0, self.width)
                y = self.rng.integers(0, self.height)
            self.cells[y, x] = player_id  # Occupy position of player

            self.players.append(
                Player(
                    player_id=player_id,
                    x=x,
                    y=y,
                    direction=self.rng.choice(directions),
                    speed=1,
                    active=True,
                )
            )
        self.controlled_player = self.players[0]  # Control first player

        return self._get_obs(self.controlled_player)

    def game_state(self):
        """Get current game state as dict."""
        return {
            'width': self.width,
            'height': self.height,
            'cells': self.cells.tolist(),
            'players': dict(p.to_dict() for p in self.players),
            'you': self.controlled_player.player_id,
            'running': sum(1 for p in self.players if p.active) > 1,
        }


class Spe_edSimulator:
    """State for the simulate function.

    Keeps a history.
    """
    def __init__(self, cells, players, rounds, changed=[], parent=None):
        self.cells = cells
        self.players = players
        self.rounds = rounds
        self.changed = changed
        self.parent = parent

    def step(self, actions):
        """Perform one simulation step"""
        return Spe_edSimulator(
            *simulate(self.cells.copy(), [p.copy() for p in self.players], self.rounds, actions),
            parent=self,
        )

    def undo(self):
        """Undo the last simulation step"""
        return self.parent

    @property
    def player(self):
        """Shorthand for the first player"""
        return self.players[0]
