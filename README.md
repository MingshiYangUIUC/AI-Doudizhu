## Experimental Doudizhu AI training with general reinforcement learning

### 1. Overview

During reinforcement learning, a lot of time is spent generating selfplay data. Here simple python modules such as multiprocessing are used to speed up selfplay.

During selfplay, tasks are assigned to CPU and GPU based on their capabilities.
- A number of game states are initialized,
- CPU prepare model inputs serially from all game states data , and send to gpu for evaluation as a batch,
- GPU evaluate the whole batch at once using the model, minimizing overhead,
- CPU uses the output from GPU to advance the game states serially, storing necessary data for training.

*So you might see as the number of game states managed by a cpu process increases, GPU utilization gets lower because it completes tasks faster than CPUs.*

In the process described above, time spent by CPU likely >> time spent by GPU (GPU is often waiting for data from CPU), so multiprocessing can be used to speed up data generation in a simple way: spawning concurrent CPU processes, all communicating with one GPU.

The major compute intensive task assigned to the CPU is search for legal actions in the available action space, which is very large in Doudizhu. c++ functions are provided to speed up this process. (Python functions will be restored if it is necessary to use them.)

### 2. Installation

Main dependencies that needs to be installed (ex. pip install numpy):
- torch
- numpy
- tqdm
- pybind11 (for using c++ functions in python)

To compile c++ functions, execute "./cpp/_compile.py" after installing above dependencies.

### 3. Usage

Scripts read parameters from a config file ".config.ini". Please rename ".config.template" to ".config.ini" and set / adjust parameters in it before running script. By default the scripts will read and use arguments in ".config.ini".

- To train a model, check \[TRAIN\] section in ".config.ini", and then run "train_main.py".

- To check performance of trained models, check \[MATCH\] section in ".config.ini", and then run "model_match_fast.py".

- To play with pretrained model, check \[PVC\] section in ".config.ini", and then run "pvc.py".

You can also call the scripts with --config "" to completely disregard ".config.ini", and use your own arguments.

### 4. Result

A small model is tested, parameter size is 11.0 MB, less than DouZero's 17.5 MB. Training is done with 1x 5900X (12 selfplay processes) + 1x 4090. On average, 650 games / 22000 actions per second are played and used for training. 

By estimate the bot reached 1550+ ELO score after 2 days. After about 10 days the model converged, and the bot reached ~1600 ELO score in Botzone. The score is on average top 1 and higher than DouZero (~1575 ELO). 

Since the model converges fairly quickly, a larger model size may enhance model performance, but that was not extensively tested. The model input is quite simple, better engineered features like that in DouZero may enhance model performance.