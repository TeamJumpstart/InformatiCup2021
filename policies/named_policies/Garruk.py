from heuristics.conditions.named_conditions import EndgameCondition, LategameCondition, MidgameCondition
from policies import ConditionalPolicy, load_named_policy

pol = ConditionalPolicy(
    policies=[
        load_named_policy("HollyV2"),
        load_named_policy("Donna"),
        load_named_policy("Elspeth"),
        load_named_policy("ElspethV2"),
    ],
    conditions=[EndgameCondition(), LategameCondition(), MidgameCondition()],
    thresholds=[True, True, True],
)
