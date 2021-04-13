from policies import ConditionalPolicy, load_named_policy
from heuristics.conditions.named_conditions import EndgameCondition, MidgameCondition, LategameCondition

pol = ConditionalPolicy(
    policies=[
        load_named_policy("HollyV2"),
        load_named_policy("DonnaV3"),
        load_named_policy("Elspeth"),
        load_named_policy("ElspethV7"),
    ],
    conditions=[EndgameCondition(), LategameCondition(), MidgameCondition()],
    thresholds=[True, True, True],
)
