from policies import RandomPolicy, SpiralPolicy, load_named_policy

# participating policies with short nick names
policies = [
    load_named_policy("Adam"),
    RandomPolicy(),
    SpiralPolicy(),
    load_named_policy("PathLength"),
    load_named_policy("Region"),
    load_named_policy("OppDist"),
]
min_size = 41  # Observed server dimensions
max_size = 80
n_players_distribution = [0.14, 0.2, 0.22, 0.22, 0.22]  # Estimated server distribution

n_games = 500
