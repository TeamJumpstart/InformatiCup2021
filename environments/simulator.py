import numpy as np
from environments.spe_ed import Player, directions
from environments.spe_ed_env import Spe_edEnv


class SimulatedSpe_edEnv(Spe_edEnv):
    def __init__(self, width, height, opponent_policies, seed=None):
        Spe_edEnv.__init__(self, width, height)

        self.opponent_policies = opponent_policies
        self.seed(seed)

    def _perform_action(self, player, action):
        if action == 'turn_left':
            player.direction = player.direction.turn_left()
        elif action == 'turn_right':
            player.direction = player.direction.turn_right()
        elif action == 'slow_down':
            if player.speed <= 1:  # Check minimum speed
                player.active = False
            else:
                player.speed -= 1
        elif action == 'speed_up':
            if player.speed >= 10:  # Check maximum speed
                player.active = False
            else:
                player.speed += 1

    def step(self, action):
        # Gather actions for all players (do this before game state changes)
        player_actions = [(self.controlled_player, action)]
        for player in self.players:  # Compute actions of opponents
            if not player.active or player == self.controlled_player:
                continue
            policy = self.opponent_policies[player.player_id - 2]
            obs = self._get_obs(player)
            player_actions.append((player, policy.act(*obs)))

        # Perform actions
        for player, action in player_actions:
            self._perform_action(player, action)

        # Move players
        newly_occupied = {}
        for player in self.players:
            if not player.active:
                continue
            pos = np.array([player.x, player.y])
            for i in range(player.speed):
                pos += player.direction.cartesian
                if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.width or pos[1] >= self.height:
                    # Player left bounds
                    player.active = False
                    break

                # Check for jumps
                if self.round % 6 == 0 and i > 0 and i < player.speed - 1:
                    continue

                if self.cells[pos[0], pos[1]] != 0:
                    # Collision
                    player.active = False
                    self.cells[pos[0], pos[1]] = -1

                    if tuple(pos) in newly_occupied:  # Occupancy is from this round
                        newly_occupied[tuple(pos)].active = False  # Other player loses, too
                else:
                    # No collision
                    self.cells[pos[0], pos[1]] = player.player_id
                    newly_occupied[tuple(pos)] = player  # Remember this cell
            player.x = pos[0]
            player.y = pos[1]

        # Round completed
        self.round += 1

        done = sum(1 for p in self.players if p.active) < 2
        reward = 1 if done and self.controlled_player.active else 0
        return self._get_obs(self.controlled_player), reward, done, {}

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Generate a new game"""
        self.round = 1
        self.cells[:] = 0  # Clear occupancies
        self.players.clear()

        # Generate players
        for i in range(len(self.opponent_policies) + 1):
            player_id = i + 1

            # Choose startion position
            x = self.rng.integers(0, self.width)
            y = self.rng.integers(0, self.height)
            while self.cells[x, y] != 0:  # Ensure chosen location is empty
                x = self.rng.integers(0, self.width)
                y = self.rng.integers(0, self.height)
            self.cells[x, y] = player_id  # Occupy position of player

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
