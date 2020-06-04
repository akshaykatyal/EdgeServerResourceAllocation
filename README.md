
# Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for Dynamic Controller Assignment in SD-IoV


This is the code for implementing the MADDPG algorithm for dynamic controller assignment in SD-IoV.
It is based on the OpenAI MADDPG code: https://github.com/openai/maddpg.git.
If you have some deployment problem, please refer to its tips.

## Installation

- Known dependencies: Python (3.7.3), tensorflow (1.12.0), numpy (1.14.5)


## Command-line options

- To train, `cd` into the `experiments` directory and run `train_controller.py`:

``python3 train_controller.py``


### Environment options

- `--scenario`: defines which environment "simple_controller".

- `--num-episodes` total number of training episodes (default: `4000`)

- `--Group-traffic` number of days (default: `"7'`)

- `--step-num`: number of time step in one day (default: `"48"`)

- `--l-type`: algorithm used for the policies in the environment (default: `"maddpg"`; options: {`"maddpg"`, `"ddpg"`})

- `--Q-type`: Queue type (options: {`"finite"`, `"inf"`})


### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `test`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `"/Restore/"`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)

- `--data-dir`: directory where data is saved

- `--vehicle-data-dir`: directory of vehicle's location data (default: `"./DATA/SDNdata_48_2/"`)

## Code structure

- `./experiments/train_controller.py`: contains code for training MADDPG on the IoV

- `./experiments/DATA/`: contains data of vehicles loaction

- `./experiments/Compare.py`: Used to plot figures

- `./maddpg/trainer/maddpg.py`: core code for the MADDPG algorithm

- `./maddpg/trainer/replay_buffer.py`: replay buffer code for MADDPG

- `./maddpg/common/distributions.py`: useful distributions used in `maddpg.py`

- `./maddpg/common/tf_util.py`: useful tensorflow functions used in `maddpg.py`

- `"./multiagent/multiagent/`": multiagent enviroment

- `"./multiagent/scenarios/simple_controller.py`": Scenario of controller in IoV
