import numpy as np
from policies.policy import Policy


class FutureStepsPolicy(Policy):
    """Policy sums number of possible steps at the current board state considering future {n_steps}.
    Tries to avoid close quarter combat with other agents.

    Baseline strategy, smarter policies should be able to outperform this.
    """

    def __init__(self, n_steps=3, seed=None):
        """Initialize FutureStepsPolicy.

        """
        self.rng = np.random.default_rng(seed)
        self.n_steps = n_steps

    def act(self, cells, player, opponents, round):

        def sum_future_steps(pos, direction, n_steps): 
            number_of_moves = 0
            pos = pos + direction.cartesian # update one position step            
            
            if cells.is_free(pos): # check current pos
                number_of_moves += 1 # valid move, add 1
            else:                
                return number_of_moves # invalid move, return

            if n_steps <= 0:
                return number_of_moves # reached n_steps in future, return
            else:
                # try to move left, add num of possible future steps                
                number_of_moves += sum_future_steps(pos, direction.turn_left(), n_steps - 1)
                # try to move forward, add num of possible future steps
                number_of_moves += sum_future_steps(pos, direction, n_steps - 1)
                # try to move right, add num of possible future steps
                number_of_moves += sum_future_steps(pos, direction.turn_right(), n_steps - 1)   
                return number_of_moves # tried all directions, return

            return 0 #end sum_future_steps

        """ Compute number of steps after `n_steps` in the future given an initial direction
        """
        sum_left = sum_future_steps(player.position, player.direction.turn_left(), self.n_steps)
        sum_forward = sum_future_steps(player.position, player.direction, self.n_steps)
        sum_right = sum_future_steps(player.position, player.direction.turn_right(), self.n_steps)

        """Choose action based on number of possible future steps.
        """

        if (sum_left > sum_forward) & (sum_left >= sum_right):
            return "turn_left"
        elif (sum_right > sum_forward) & (sum_right >= sum_left):
            return "turn_right"
        else:
            return "change_nothing"