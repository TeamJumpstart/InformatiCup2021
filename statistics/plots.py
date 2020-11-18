import matplotlib.pyplot as plt
from pathlib import Path
from visualization import WinRateAx
from statistics.stats import fetch_statistics


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
