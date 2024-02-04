from policies import SpiralPolicy, load_named_policy

# participating policies with short nick names
policies = [
    load_named_policy("AdamV2"),
    load_named_policy("AdamV2+V"),
    SpiralPolicy(),
    load_named_policy("Region"),
    load_named_policy("Clark"),
]
min_size = 41  # Observed server dimensions
max_size = 80
n_players_distribution = [0.14, 0.2, 0.22, 0.22, 0.22]  # Estimated server distribution

n_games = 500

write_logs = False
