from heuristics.conditions.named_conditions.endgame_condition import EndgameCondition
from policies import ConditionalPolicy, EndgamePolicy, load_named_policy

ConditionalPolicy(
    [EndgamePolicy(), load_named_policy("AdamV2")],
    conditions=[EndgameCondition()],
    thresholds=[0.9],
)
