from heuristics import EndgameConditionHeuristic
from policies import ConditionalPolicy, EndgamePolicy, load_named_policy

ConditionalPolicy(
    [EndgamePolicy(), load_named_policy("AdamV3")],
    conditions=[EndgameConditionHeuristic()],
    thresholds=[0.9],
)
