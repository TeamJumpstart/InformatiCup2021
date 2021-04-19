<div align="center">
  <img src="images/TeamJumpstart.png" alt="Team Jumpstart" width="50%"/>
</div>

This project is the contribution of [Team Jumpstart](https://teamjumpstart.github.io/) to the [InformatiCup2021 challenge](https://raw.githubusercontent.com/informatiCup/InformatiCup2021/master/call_for_participation_a4.pdf) provided by the [German Informatics Society (GI)](https://gi.de/).
We offer our own AI agent that is able to competitively play the game [**spe_ed**](https://github.com/InformatiCup/InformatiCup2021/blob/master/spe_ed.pdf).

The accompanying [paper](https://github.com/TeamJumpstart/InformatiCup2021/releases/download/v1.0.0-submission/Informaticup2021.Theoretische.Ausarbeitung.pdf) documents our approach.


<figure class="video_container">
  <video src="images/Jumpstart_teaser.mp4"></video>
</figure>

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

![Simulator](/images/simulator.png)

**Renderer**: render games from log files to video format.

```sh
docker run -d --name=spe_ed spe_ed --sim --show
```

![Renderer](/images/renderer.png)

**Plot mode**: create plots for logged games from the webserver. This is intended to observe player behavior as it gives an overview over all tracked games.

```sh
docker run -d --name=spe_ed spe_ed plot --log-dir .\logs\
```

![Opponents Scatterplot](/images/game-history2.png)

**Tournament mode**: run a tournament of multiple policies, where each game has different policy and parameter combinations. The chosen policies and parameter options like grid size can be configured in a separate tournament config file (`tournament\tournament_config.py`).

```sh
docker run -d --name=spe_ed spe_ed tournament --log-dir .\tournament\logs\
```

**Tournament plot mode**: create plots and matchup statistics for a played tournament given by its logs.

```sh
docker run -d --name=spe_ed spe_ed tournament-plot --log-dir .\tournament\logs\
```

![Matchup Results](/images/matchups.png)