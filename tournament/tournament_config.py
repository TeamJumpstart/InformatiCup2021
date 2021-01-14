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
# different map sizes to be played
width_height_pairs = [
    (30, 30), (50, 50)
]  # or use randomly generated ranges: [(randint(20,51), randint(20,51)), (randint(50, 101), randint(50, 101))]
# number of games to be played for each constellation
number_games = 2
