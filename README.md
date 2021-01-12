# InformatiCup2021

This project is the contribution of Team Jumpstart to the InformatiCup2021 challenge provided by the GI.
We offer our own AI agent that is able to independently play the game **spe_ed** (https://github.com/InformatiCup/InformatiCup2021/blob/master/spe_ed.pdf).

## Usage

Build the docker image:
```
docker build -t spe_ed .
```

Run the docker image in the default mode, i.e. seeking a connection with a provided websocket server:
```
docker run -e URL="wss://msoll.de/spe_ed" -e KEY="your_key" spe_ed
```
This will start a websocket client, that connects to the server and selects an action, whenever a new state is transmitted by the server.


### Extensions

Can be build with the same docker image.
Other modes than the default "play on websocket server" can be selected.

**Simulator**: play in a simulator environment instead of the server. 
```
docker run -d --name=spe_ed spe_ed --sim --show
```

**Plot mode**: creates plots for logged games from the webserver. This is intended to observe player behavior as it gives an overview over all tracked games.
```
docker run -d --name=spe_ed spe_ed plot --log-dir .\logs\
```

**Tournament mode**: runs a tournament of multiple policies, where each game has different policy and parameter combinations. The chosen policies and parameter options like grid size can be configured in a separate tournament config file (``tournament\tournament_config.py``).
```
docker run -d --name=spe_ed spe_ed tournament --log-dir .\tournament\logs\
```

**Tournament plot mode**: creates plots and matchup statistics for a played tournament given by its logs.
```
docker run -d --name=spe_ed spe_ed tournament-plot --log-dir .\tournament\logs\
```
