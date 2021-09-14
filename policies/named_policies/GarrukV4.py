from heuristics.conditions.named_conditions import EndgameCondition, LategameCondition, MidgameCondition
from policies import ConditionalPolicy, load_named_policy

# Variant of GarrukV3 with closing_iterations working
pol = ConditionalPolicy(
    policies=[
        load_named_policy("HollyV2"),
        load_named_policy("DonnaV3"),
        load_named_policy("ElspethV3"),
        load_named_policy("ElspethV9"),
    ],
    conditions=[EndgameCondition(), LategameCondition(), MidgameCondition()],
    thresholds=[True, True, True],
)
