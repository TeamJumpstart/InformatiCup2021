import numpy as np
from policies.policy import Policy


class FutureStepsPolicy(Policy):
    """Policy sums number of possible steps at the current board state considering future {n_steps}.
    Tries to avoid close quarter combat with other agents.

    Baseline strategy, smarter policies should be able to outperform this.
    """
    def __init__(self, n_steps=3, dynamic=False, seed=None):
        """Initialize FutureStepsPolicy.

        Args:
            n_steps: Number of steps to predict into the future.
            dynamic: If `True` considers the predicted steps into account for future moves, e.g. updates the cell state for the agent itself,
                     if `False` does not update the cell state after a move and only considers the initial static board state for each prediction step.
            seed: Seed for the random number generator. Use a fixed seed for reproducibility,
                  or pass `None` for a random seed.
            p: Propability weights of each action. Pass `None` for uniform distribution.

        """
        self.n_steps = n_steps
        self.dynamic = dynamic
        self.rng = np.random.default_rng(seed)

    def act(self, cells, player, opponents, round):
        def sum_future_steps(pos, direction, n_steps, future_steps=None):
            number_of_moves = 0
            pos = pos + direction.cartesian  # update one position step

            if cells.is_free(pos):  # check current pos
                if self.dynamic:  # check dynamic board state with future moves
                    if np.any(future_steps == pos):
                        return number_of_moves
                    else:
                        np.append(future_steps, pos, axis=-1)
                        number_of_moves += 1
                else:  # assume a static board state
                    number_of_moves += 1  # valid move, add 1
            else:
                return number_of_moves  # invalid move, return

            if n_steps <= 0:
                return number_of_moves  # reached n_steps in future, return
            else:
                # try to move left, add num of possible future steps
                number_of_moves += sum_future_steps(pos, direction.turn_left(), n_steps - 1, future_steps)
                # try to move forward, add num of possible future steps
                number_of_moves += sum_future_steps(pos, direction, n_steps - 1, future_steps)
                # try to move right, add num of possible future steps
                number_of_moves += sum_future_steps(pos, direction.turn_right(), n_steps - 1, future_steps)
                return number_of_moves  # tried all directions, return

            return 0  # end sum_future_steps

        # Compute number of steps after `n_steps` in the future given an initial direction
        sum_left = sum_future_steps(player.position, player.direction.turn_left(), self.n_steps, [])
        sum_forward = sum_future_steps(player.position, player.direction, self.n_steps, [])
        sum_right = sum_future_steps(player.position, player.direction.turn_right(), self.n_steps, [])

        # Choose action based on number of possible future steps.
        if (sum_left > sum_forward) & (sum_left >= sum_right):
            return "turn_left"
        elif (sum_right > sum_forward) & (sum_right >= sum_left):
            return "turn_right"
        else:
            return "change_nothing"
