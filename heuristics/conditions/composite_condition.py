from heuristics.conditions.condition import Condition
import numpy as np


class CompositeCondition(Condition):
    """Allows to combine multiple conditions into a single score evaluating the same board state."""
    def __init__(self, conditions, thresholds=None, logical_op=np.logical_and, compare_op=np.greater_equal):
        """Initialize CompositeCondition.

        Args:
            heuristics: An array containing different `Conditions` which should be evaluated in combination.
            thresholds: Threshold of each `Condition`. Pass `None` for binary thresholds.
            logical_op: Logical operator function to combine conditions. default: np.logical_and
            compare_op: Compare operator to compare the condition score with the threshold.
        """
        self.conditions = conditions
        if thresholds is not None and len(thresholds) != len(conditions):
            raise ValueError(f"Number of weights {thresholds} does mot match number of heuristics {conditions}")
        if thresholds is None:
            self.thresholds = np.ones(len(conditions))
        else:
            self.thresholds = thresholds
        self.logical_op = logical_op
        self.compare_op = compare_op

    def score(self, cells, player, opponents, rounds):
        """Compute the combined condition score."""
        score = True
        for condition, threshold in zip(self.conditions, self.thresholds):
            score = self.logical_op(
                score, self.compare_op(condition.score(cells, player, opponents, rounds), threshold)
            )

        return score

    def __str__(self):
        """Get readable representation."""
        return "CompositeCondition(" + \
            f"[{','.join([str(condition) for condition in self.conditions])}], " + \
            f"weights={self.thresholds}, " + \
            f"logical_op={self.logical_op}, " + \
            ")"
