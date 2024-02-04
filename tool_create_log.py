"""This tool script can be used to create custom log files."""

import json

import numpy as np

from environments import Spe_edSimulator
from environments.spe_ed import Player, directions_by_name


def write_history(log_file, sim):
    # Compute board states (backtrack to initial state)
    states = []
    running = False
    while sim is not None:
        states = [
            {
                "width": sim.cells.shape[1],
                "height": sim.cells.shape[0],
                "cells": sim.cells.tolist(),
                "players": dict(p.to_dict() for p in sim.players),
                "you": sim.players[0].player_id,
                "running": running,
            }
        ] + states
        sim = sim.parent
        running = True

    # Export saved game
    with open(log_file, "w") as f:
        json.dump(states, f, indent=4, sort_keys=True)


# Step 1: Initialize sim
cells = np.zeros((5, 5), dtype=int)
players = [Player(player_id=1, x=2, y=2, direction=directions_by_name["right"], speed=1, active=True)]
# Occupy initial cells
for p in players:
    cells[p.y, p.x] = 1
sim = Spe_edSimulator(cells, players, rounds=1)

# Step 2: Peform actions
sim = sim.step(["change_nothing"])
sim = sim.step(["turn_right"])

# Step 3: Export
write_history("test.json", sim)
