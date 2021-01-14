from policies.policy import Policy


class ConditionalPolicy(Policy):
    """Allows to evaluate different policies in a hierarchical manner.

    The first policy, which whose condition satisfies their corresponding threshold, is executed.
    """
    def __init__(self, policies, conditions, thresholds):
        """Initialize ConditionalPolicy. If given only one policy, conditions and thresholds can be 'None'.

        Args:
            policies: List of 'Policy's. Provide at least one policy to execute.
            conditions: List of 'Heuristic's. Should be a list with lenght one less than policies.
            thresholds: List of floats. Length should match conditions.
        """
        self.policies = policies
        if policies is None:
            raise ValueError(f"No policies provided {str(policies)}.")

        if len(policies) == 1:
            self.conditions = []
            self.thresholds = []
        else:
            self.conditions = conditions
            if conditions is None or len(policies) != len(conditions) + 1:
                raise ValueError(
                    f"Number of policies {str(policies)} should be larger by one\
                     than number of conditions {str(conditions)}."
                )
            self.thresholds = thresholds
            if thresholds is None or len(conditions) != len(thresholds):
                raise ValueError(
                    f"Number of conditions {str(conditions)} does mot match number of thresholds {str(thresholds)}."
                )

    def act(self, cells, player, opponents, rounds):
        """Execute the first policy, whose condition satisfies its threshold."""
        for policy, condition, threshold in zip(self.policies, self.conditions, self.thresholds):
            if condition.score(cells, player, opponents, rounds) >= threshold:
                return policy.act(cells, player, opponents, rounds)
        return self.policies[-1].act(cells, player, opponents, rounds)

    def __repr__(self):
        """Get exact representation."""
        return "ConditionalPolicy(" + \
            f"policies={str(self.policies)}, " + \
            f"conditions={str(self.conditions)}, " + \
            f"thresholds={self.thresholds}, " + \
            ")"
