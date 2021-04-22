# InformatiCup2021

![Tests Passes](https://github.com/TeamJumpstart/InformatiCup2021/actions/workflows/docker-build.yml/badge.svg)

<div align="center">
  <img src="images/TeamJumpstart.png" alt="Team Jumpstart" width="50%"/>
</div>

This project is the contribution of [Team Jumpstart](https://teamjumpstart.github.io/) to the [InformatiCup2021 challenge](https://raw.githubusercontent.com/informatiCup/InformatiCup2021/master/call_for_participation_a4.pdf) provided by the [German Informatics Society (GI)](https://gi.de/).
We offer our own AI agent that is able to competitively play the game [**spe_ed**](https://github.com/InformatiCup/InformatiCup2021/blob/master/spe_ed.pdf).

The accompanying [paper](https://github.com/TeamJumpstart/InformatiCup2021/releases/download/v1.0.0-submission/Informaticup2021.Theoretische.Ausarbeitung.pdf) documents our approach.

The challenge lies in both finding a strategy that ensures the own agent's survival, and at the same time anticipating the other player's future movements. With up to six players concurrently playing, there is a multitude of variables to consider, as well as added complexity with varying speed and the ability to jump. To address these issues, we propose [depth-limited heuristic search](policies/action_search.py) and [Monte-Carlo based heuristic search](heuristics/randomprobing_heuristic.py) algorithms. We approximate the board state via heuristics to choose actions within a limited search horizon and incorporate an early-out mechanism to meet critical deadlines. The agent will explore potential future actions based on different heuristics, while simultaneously predicting the probability of opponent movements to choose the best immediate action.

Animations of several heuristics are shown on the [project website](https://teamjumpstart.github.io/InformatiCup2021/).

<div align="center">
  <a href="https://teamjumpstart.github.io/InformatiCup2021/">
    <img src="images\Jumpstart_teaser.gif" alt="Team Jumpstart"/>
  </a>
</div>

## Usage

Clone the repo and build the docker image:

```sh
docker build -t spe_ed .
```

Run the docker image in the default mode, i.e. seeking a connection with a provided websocket server:

```sh
docker run -e URL="wss://msoll.de/spe_ed" -e KEY="your_key" -e TIME_URL="https://msoll.de/spe_ed_time" spe_ed
```

This will start a websocket client, that connects to the server and selects an action, whenever a new state is transmitted by the server.

```text
usage: main.py [-h] [--show] [--render-file RENDER_FILE] [--sim] [--log-file LOG_FILE] [--log-dir LOG_DIR] [--t-config T_CONFIG] [--upload] [--fps FPS] [--cores CORES]
               [{play,replay,render_logdir,plot,tournament,tournament-plot}]

spe_ed

positional arguments:
  {play,replay,render_logdir,plot,tournament,tournament-plot}

optional arguments:
  -h, --help            show this help message and exit
  --show                Display games using an updating matplotlib plot.
  --render-file RENDER_FILE
                        File to render to. Should end with .mp4
  --sim                 The simulator environment runs a local simulation of Spe_ed instead of using the webserver.
  --log-file LOG_FILE   Path to a log file, used to load and replay games.
  --log-dir LOG_DIR     Directory for storing or retrieving logs.
  --t-config T_CONFIG   Path of the tournament config file containing which settings to run.
  --upload              Upload generated log to cloud server.
  --fps FPS             FPS for rendering.
  --cores CORES         Number of cores for multiprocessing, default uses all.
```

### Extensions

Can be build with the same docker image.
Other modes than the default "play on websocket server" can be selected.

**Simulator**: play in a simulator environment instead of the server and view the played game in a matplotlib plot.

```sh
docker run -d --name=spe_ed spe_ed --sim --show
```

<div align="center">
  <img src="/images/simulator.png" alt="Simulator" style="max-width: 100%;"/>
</div>

**Renderer**: render games from log files to video format.

```sh
docker run -d --name=spe_ed spe_ed --sim --show
```

<div align="center">
  <img src="/images/renderer.png" alt="Renderer" style="max-width: 100%;"/>
</div>

**Plot mode**: create plots for logged games from the webserver. This is intended to observe player behavior as it gives an overview over all tracked games.

```sh
docker run -d --name=spe_ed spe_ed plot --log-dir .\logs\
```

<div align="center">
  <img src="/images/game-history2.png" alt="Opponents Scatterplot" style="max-width: 100%;"/>
</div>

**Tournament mode**: run a tournament of multiple policies, where each game has different policy and parameter combinations. The chosen policies and parameter options like grid size can be configured in a separate tournament config file (`tournament/tournament_config.py`).

```sh
docker run -d --name=spe_ed spe_ed tournament --log-dir tournament/logs
```

**Tournament plot mode**: create plots and matchup statistics for a played tournament given by its logs.

```sh
docker run -d --name=spe_ed spe_ed tournament-plot --log-dir tournament/logs
```

<div align="center">
  <img src="/images/matchups.png" alt="Matchup Results" style="max-width: 100%;"/>
</div>
