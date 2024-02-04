from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def easeInOutCubic(t):
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def easeInOutSine(t):
    return -(np.cos(np.pi * t) - 1) / 2


def easeOutCubic(t):
    return 1 - (1 - t) ** 3


def names_to_codes(names):
    code_map = {}
    id = 0

    codes = []
    for name in names:
        if name not in code_map:
            code_map[name] = id
            id += 1

        codes.append(code_map[name])

    return codes


def animate_opponent_sequence(stats_file, output_dir, total_rows, row_offset, row_height, variance=1):
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(stats_file, parse_dates=["date"]).sort_values("date")

    df["names"] = names_to_codes(df["names"])
    df["date"] -= pd.to_datetime("2020-11-01 01:11:47")
    df["date"] /= pd.to_timedelta("50 days 22:44:16")

    plt.figure(figsize=(17.5, total_rows * row_height), dpi=128)
    scatter = plt.scatter(df["date"], df["names"], c=[(0.3, 0.3, 0.3)], alpha=0.2, linewidths=0)

    plt.xlim(-0.01, 1.01)
    plt.ylim(-5, total_rows + 5 - 1)
    plt.axis("off")
    plt.tight_layout(pad=0)

    x = np.array(df["date"])
    y = df["names"].max() - 1 - np.array(df["names"])

    T = 50
    fade_in = 5
    rnd = np.random.RandomState(0).normal(size=len(x)) * variance

    anim_offset = -x * (T - 1 - fade_in) + y + rnd

    anim_offset -= np.max(anim_offset)

    anim_offset *= -(T - 1 - fade_in) / np.min(anim_offset)

    for frame, t in tqdm(enumerate(np.linspace(0, 1, T)), total=T):
        data = scatter.get_offsets()
        alpha = np.empty(len(x))
        for i in range(len(data)):
            p = easeInOutSine(t) * (T - 1) + anim_offset[i]

            if p < 0:
                data[i, 1] = np.nan
                alpha[i] = 0
            else:
                data[i, 1] = ((y[i]) + row_offset) * min(1, easeOutCubic(p / fade_in))
                alpha[i] = 0.2 * min(1, easeOutCubic(p / fade_in))

        scatter.set_alpha(alpha)

        plt.savefig(output_dir / f"{frame:04}.png", transparent=True)

    plt.close()


animate_opponent_sequence(
    stats_file=r"F:\spe_ed\statistics_bots.csv",
    output_dir=Path(r"F:\spe_ed2\render\bots"),
    total_rows=78,
    row_offset=69,
    row_height=1 / 8,
    variance=1 / 2,
)

animate_opponent_sequence(
    stats_file=r"F:\spe_ed\statistics_loggers.csv",
    output_dir=Path(r"F:\spe_ed2\render\loggers"),
    total_rows=78,
    row_offset=62,
    row_height=1 / 8,
    variance=1,
)

animate_opponent_sequence(
    stats_file=r"F:\spe_ed\statistics_opponents.csv",
    output_dir=Path(r"F:\spe_ed2\render\opponents"),
    total_rows=78 * 2,
    row_offset=0,
    row_height=1 / 16,
    variance=1 / 2,
)
