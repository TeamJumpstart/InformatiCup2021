import matplotlib.pyplot as plt
from pathlib import Path
from visualization import WinRateAx
from statistics.stats import fetch_statistics, get_win_rate, create_matchup_stats
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


def plot_win_rate(policy_names, policy_nick_names, stats, output_file, number_of_players=None, grid_size=None):
    win_rates = []
    for policy_name in policy_names:
        win_rates.append(get_win_rate(policy_name, stats, number_of_players=number_of_players, grid_size=grid_size))

    plt.figure()
    plt.bar(policy_nick_names, win_rates)
    plt.xlabel("Policy name")
    plt.ylabel("Win rate")

    plt.savefig(output_file)
    plt.close()


def create_tournament_plots(log_dir, stats_dir):
    # Load statistics
    stats = fetch_statistics(log_dir, stats_dir / "statistics.csv", tournament_mode=True)

    # Create output folder
    plot_dir = stats_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create plots of matchup stats, overall win rate of each policy
    policy_nick_names = np.unique(np.concatenate([matchup.split('_')[:-2] for matchup in stats['matchup'].values]).flat)
    policy_names = np.unique(np.concatenate(stats['names'].values).flat)
    create_matchup_stats(stats, stats_dir / "matchup_statistics.csv")
    print(policy_nick_names)

    plot_win_rate(policy_names, policy_nick_names, stats, plot_dir / "win_rate.png")
    plot_win_rate(policy_names, policy_nick_names, stats, plot_dir / "win_rate_small.png", grid_size=(30, 30))
    plot_win_rate(policy_names, policy_nick_names, stats, plot_dir / "win_rate_big.png", grid_size=(50, 50))
    plot_win_rate(policy_names, policy_nick_names, stats, plot_dir / "win_rate_1v1.png", number_of_players=2)
    plot_win_rate(policy_names, policy_nick_names, stats, plot_dir / "win_rate_6p.png", number_of_players=6)
