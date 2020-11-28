import matplotlib.pyplot as plt
from pathlib import Path
from visualization import WinRateAx
from statistics.stats import fetch_statistics
import numpy as np


def plot_win_rate_over_time(output_file, stats):
    """Plot our win rate over time."""
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    stats = stats[stats['date'] >= '2020-10-30']  # Truncate early data
    won = stats['you'] == stats['winner']
    WinRateAx(fig, ax, stats["date"], won)

    ax.legend(loc="lower right")
    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def create_plots(log_dir, stats_file):
    # Load statistics
    stats = fetch_statistics(log_dir, stats_file)

    # Create output folder
    plot_dir = Path(stats_file).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create plots
    plot_win_rate_over_time(plot_dir / "win_rate.pdf", stats)


def plot_win_rate(policy_names, stats, output_file):
    won_games = []
    for policy_name in policy_names:
        relevant_games = stats[stats["matchup"].str.contains(policy_name, regex=False)]
        won_games.append(
            len([winner for winner in relevant_games["winner"] if winner is not None and winner == policy_name]) / # does not work because winner doesnt have policy name
            len(relevant_games)
        )

    plt.figure()
    plt.bar(policy_names, won_games)
    plt.xlabel("Policy name")
    plt.ylabel("Win rate")

    plt.savefig(output_file)
    plt.close()


def create_tournament_plots(log_dir, stats_file):
    # Load statistics
    stats = fetch_statistics(log_dir, stats_file, tournament_mode=True)

    # Create output folder
    plot_dir = Path(stats_file).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create plots of matchup stats, overall win rate of each policy
    policy_names = np.unique(np.concatenate([matchup.split('_')[:-1] for matchup in stats['matchup'].values]).flat)
    print(policy_names)
    plot_win_rate(policy_names, stats, plot_dir / "win_rate.png")
