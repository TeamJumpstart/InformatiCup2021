from pathlib import Path
from statistics.stats import fetch_statistics, get_win_rate, normalize_winrate

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization import WinRateAx


def plot_win_rate_over_time(output_file, stats):
    """Plot our win rate over time."""
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    stats = stats[stats["date"] >= "2020-10-30"]  # Truncate early data
    won = stats["you"] == stats["winner"]
    WinRateAx(fig, ax, stats["date"], won)

    ax.legend(loc="lower right")
    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def plot_n_players(output_file, stats, cutoff_date=None):
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    if cutoff_date is not None:
        stats = stats[stats["date"] >= cutoff_date]

    n_players = stats["names"].apply(len)

    played = np.array([(n_players == n).agg(["mean", "count", "std"]) for n in range(2, 7)])
    mean, count, std = played[:, 0], played[:, 1], played[:, 2]

    # Compute confidence interval
    conf = 1.96 * std / np.sqrt(count)

    plt.bar(np.arange(2, 7), mean, yerr=conf)
    ax.axhline(1 / 5, color="black", linestyle="dashed", label="uniform distribution")

    plt.xlabel("number of players")
    plt.ylabel("ratio of logged games")
    ax.legend(loc="lower right")
    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def plot_grid_size(output_file, stats):
    plt.figure(figsize=(6.4 * 2, 4.8))

    plt.subplot(1, 2, 1)
    plt.hist(stats["width"], density=True)
    plt.xlabel("width")
    plt.ylabel("density")

    plt.subplot(1, 2, 2)
    plt.hist(stats["height"], density=True)
    plt.xlabel("height")
    plt.ylabel("density")

    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def plot_grid_size_correlation(output_file, stats, include_n_players=False):
    data = stats[["width", "height"]].copy()
    if include_n_players:
        data["n_players"] = stats["names"].apply(len)

    sns.pairplot(data, kind="hist", diag_kws={"bins": 20}, plot_kws={"bins": 20})

    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def plot_competition(output_file, stats_dir, stats):
    you_aliases = stats["you"].unique()
    known_bots = set(pd.read_csv(stats_dir / "known_bots.csv")["known_bots"])

    # Explode names
    stats = stats[["date", "winner", "names"]].explode("names")

    # Player type
    stats["type"] = "opponent"
    stats.loc[stats["names"].isin(known_bots), "type"] = "bot"
    stats.loc[stats["names"].isin(you_aliases), "type"] = "we"

    # Winning
    stats["won"] = stats["winner"] == stats["names"]

    plt.figure(figsize=(10, 5))

    ax = sns.scatterplot(data=stats, x="date", y="names", style="type", hue="won", markers="^s", zorder=1000)  # o
    plt.grid(zorder=-1000)

    ax.get_yaxis().set_ticks([])
    ax.set_ylabel("player name")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def create_plots(log_dir, stats_file):
    # Load statistics
    stats = fetch_statistics(log_dir, stats_file)

    # Create output folder
    plot_dir = Path(stats_file).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create plots
    for ext in ("png", "pdf"):
        plot_competition(plot_dir / f"competition.{ext}", stats_file.parent, stats[stats["date"] >= "2021-01-02"])
        plot_win_rate_over_time(plot_dir / f"win_rate.{ext}", stats)
        plot_n_players(plot_dir / f"n_players.{ext}", stats)
        plot_n_players(plot_dir / f"n_players_2021.{ext}", stats, cutoff_date="2021-01-01")
        plot_grid_size(plot_dir / f"grid_size.{ext}", stats)
        plot_grid_size_correlation(plot_dir / f"grid_size_corr.{ext}", stats)
        plot_grid_size_correlation(plot_dir / f"grid_size_n_players_corr.{ext}", stats, include_n_players=True)


def plot_tournament_win_rates(
    policy_names, policy_nick_names, stats, output_file, number_of_players=None, grid_size=None
):
    win_rates = np.array(
        [
            get_win_rate(policy_name, stats, number_of_players=number_of_players, grid_size=grid_size)
            for policy_name in policy_names
        ]
    )
    # Order by win rate descendingly
    order = np.argsort(win_rates[:, 0])[::-1]
    policy_nick_names = np.array(policy_nick_names)[order]
    win_rates = win_rates[order]

    mean, count, std = win_rates[:, 0], win_rates[:, 1], win_rates[:, 2]

    # Compute confidence interval
    conf = 1.96 * std / np.sqrt(count)

    plt.figure()
    plt.bar(policy_nick_names, mean, yerr=conf)
    plt.xlabel("policy")
    plt.ylabel("win rate")
    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def plot_matchups(policy_names, policy_nick_names, stats, output_file):
    """Create a matchup table.

    Args:
            policy_names: full name of policy
            policy_nick_names: policy nick names
            stats: pandas df of statistics from fetch_statistics
            output_file: File to save figure to
    """
    matchup_win_rates = []
    for policy_name in policy_names:
        policy_matchups = [
            get_win_rate(policy_name, stats, matchup_opponent=opponent_policy) for opponent_policy in policy_names
        ]
        matchup_win_rates.append([x[0] if x is not None else np.nan for x in policy_matchups])
    matchup_win_rates = np.array(matchup_win_rates)

    baseline_winrate = np.mean(1 / stats["names"].map(len))

    ax = sns.heatmap(
        normalize_winrate(matchup_win_rates, baseline_winrate),
        mask=np.isnan(matchup_win_rates),
        vmin=0,
        vmax=1,
        xticklabels=policy_nick_names,
        yticklabels=policy_nick_names,
        square=True,
        cmap="seismic",
        cbar_kws={"label": "normalized win rate"},
    )
    ax.set_title("vs.", y=1, x=-0.05)
    ax.set_facecolor("black")
    plt.yticks(va="center")
    plt.tick_params(axis="both", which="major", labelbottom=False, bottom=False, top=False, labeltop=True, left=False)
    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def plot_policy_times(stats_dir, output_file):
    sns.boxplot(x="policy", y="time", data=pd.read_csv(stats_dir / r"times.csv"))
    plt.tight_layout(pad=0)

    plt.savefig(output_file)
    plt.close()


def create_tournament_plots(log_dir, stats_dir):
    # Load statistics
    stats = fetch_statistics(log_dir, stats_dir / "statistics.csv", key_column="uuid")

    # Create output folder
    plot_dir = stats_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Create plots of matchup stats, overall win rate of each policy
    policy_names = np.unique(np.concatenate(stats["names"].values).flat)
    policy_nick_names = [name if len(name) <= 10 else name[:8] + "..." for name in policy_names]

    plot_matchups(policy_names, policy_nick_names, stats, plot_dir / "matchups.png")

    plot_tournament_win_rates(policy_names, policy_nick_names, stats, plot_dir / "win_rate.png")
    plot_tournament_win_rates(
        policy_names, policy_nick_names, stats, plot_dir / "win_rate_small.png", grid_size="small"
    )
    plot_tournament_win_rates(
        policy_names, policy_nick_names, stats, plot_dir / "win_rate_medium.png", grid_size="medium"
    )
    plot_tournament_win_rates(
        policy_names, policy_nick_names, stats, plot_dir / "win_rate_large.png", grid_size="large"
    )
    for p in range(2, min(7, len(policy_names) + 1)):
        plot_tournament_win_rates(
            policy_names, policy_nick_names, stats, plot_dir / f"win_rate_{p}p.png", number_of_players=p
        )

    plot_policy_times(stats_dir, plot_dir / "policy_times.png")
