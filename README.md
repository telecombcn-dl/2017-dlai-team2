# 2017-dlai-team2
DLAI 2017 Team 2

## Introduction

This project is based on [NeuralKart](https://github.com/rameshvarun/NeuralKart), which has been extended to use the minimap as input into a separate convolutional neural network as well as the introcution of LSTM layers to make current predictions based not only on the current input, but also on the past. 

## Set up
As this project is based on [NeuralKart](https://github.com/rameshvarun/NeuralKart), it has the same dependencies. Simulations have been carried out using [BizHawk](https://github.com/TASVideos/BizHawk) emulator and Python 3.

### Python Dependencies
The following Python dependencies need to be installed.

- Tensorflow
- Keras
- Pillow
- mkdir_p

## Getting started
### Directory structure

- **/Lua contains** all lua and python scripts needed
- **/Lua/lualibs** contains necessary helper methods for the main lua scripts
- **/Lua/weights** contains the weight files for the NN for different tracks we've trained it on

### Files
#### Lua scripts
- **searchAIbeam_new.lua** is a script searching for the optimal path given the three best starting points of the previous iteration. Once a track has been completed, you've obtained the initial labeled dataset for a track.
- **PalyAndSearch_new.lua** is used to let the NN play for 130 frames before the AI searches for the optimal path again, providing new frames for training. Before this script can be run, the predict server has to be up and running.
- **Play.lua** is used to let the NN play all by itself, once again the prediction server needs to be up and running.

#### Python scripts
- **train.py** the original training script from [NeuralKart](https://github.com/rameshvarun/NeuralKart) which we'll use to compare our implementation against it.
- **train_new_map.py** is our implementation of the NN which makes use of the minimap and past frames.
- **predict_server.py** is the predict server, which is used to play in combination with the corresponding lua scripts.

## The neural network
<Image here>
  
## Performance
<Performance here>
