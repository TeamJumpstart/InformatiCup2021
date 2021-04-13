from policies import ConditionalPolicy, EndgamePolicy, load_named_policy
from heuristics import EndgameConditionHeuristic

ConditionalPolicy(
    [EndgamePolicy(), load_named_policy("AdamV3")],
    conditions=[EndgameConditionHeuristic()],
    thresholds=[0.9],
)
