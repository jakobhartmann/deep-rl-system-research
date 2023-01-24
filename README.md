# Deep Reinforcement Learning for Systems Research

## Introduction
This repository contains the code for my open-source project "Deep Reinforcement Learning for Systems Research", which was part of the module "R244: Large-scale data processing and optimisation" in Michaelmas term 2022 at the University of Cambridge. The project explored how researchers and practioners can utilize the open-source libraries Park and RLlib for systems and network research. Park takes a similar approach to [Gymasium](https://github.com/Farama-Foundation/Gymnasium) and provides several systems-based environments to test new reinforcement learning algorithms. An introduction to the library can be found in the following paper:

> Hongzi Mao et al. “[Park: An Open Platform for Learning Augmented Computer Systems](https://proceedings.neurips.cc/paper/2019/hash/f69e505b08403ad2298b9f262659929a-Abstract.html)”. In: Advances in Neural Information Processing Systems. Ed. by H. Wallach et al. Vol. 32. Curran Associates, Inc., 2019.

RLlib is a library built on top of the distributed execution engine [Ray](https://github.com/ray-project/ray) and can be used to design, implement, train, optimize and test existing as well as novel deep reinforcement learning algorithms. The following paper explains RLlib's main underlying ideas and concepts:

> Eric Liang et al. “[RLlib: Abstractions for Distributed Reinforcement Learning](https://proceedings.mlr.press/v80/liang18b.html)”. In: Proceedings of the 35th International Conference on Machine Learning. Ed. by Jennifer Dy and Andreas Krause. Vol. 80. Proceedings of Machine Learning Research. PMLR, July 2018, pp. 3053–3062.

As part of the open-source project, I extended the [main Park repository](https://github.com/park-project/park) in the following ways:
- Added support for [Dict](./park/spaces/dict.py) spaces in Park
- Implemented a custom `ProcessScheduling` environment using [Park](./park/envs/process_scheduling/process_scheduling_env.py) and [Gym](./process_scheduling_gym/process_scheduling_gym_env.py)
- Implemented three baseline scheduling agents for the `ProcessScheduling` environment in [Park](./process_scheduling_park_agents.py) and [Gym](./process_scheduling_gym/process_scheduling_gym_agents.py)
- Extended an [existing](https://github.com/park-project/park/blob/b533ba368881e5424fbe22280b8a86eb5a3e613c/algorithms/agent_wrapper.py) Park -> Gym space/environment [wrapper](./env_wrapper.py) to Dict, Tuple and Graph spaces
- Implemented a `test_environments()` [function](./env_wrapper.py) to automatically check which Park environments can be successfully initialized, wrapped in a Gym environment and are compatible with RLlib
- Added experimental code for optimizing hyperparameters of deep reinforcement learning algorithms using RLlib for [Park](./rllib_experiments/park.py) and [CompilerGym](./rllib_experiments/compiler_gym.py) environments



## Overview of Park Environments
Park environments can be divided into two distinct groups: real and simulated ones. Real environments are executed directly on the user's system, whereas simulated ones rely on, for example, traces of previous environment observations. However, not all (real) Park environments can be executed on each hardware device / operating system. Furthermore, Park is also not actively maintained anymore, which results in additional issues for some environments.

Another problem is that although Park exposes a similar interface to the one popularized by OpenAI Gym, it is not built on top of Gymnasium. Instead Park environments utilize custom spaces, resulting in compatibility issues with many other libraries such as RLlib. While some environments can be made compatible by converting the Park spaces to Gym spaces, others can't. The table below provides an overview of which Park environments can be initialized successfully, converted to a Gym environment and are compatible with RLlib. The tests were performed using MacOS Ventura 13.1 and Ubuntu 20.04.5 LTS.

|           Environment           |                  ID                 | Instantiation Possible | Gymnasium Compatibility | RLlib Compatibility |
|:-------------------------------:|:-----------------------------------:|:----------------------:|:-----------------------:|:-------------------:|
|     Adaptive video streaming    |                 abr                 |           :x:          |            -            |          -          |
|     Adaptive video streaming    |               abr_sim               |           :x:          |            -            |          -          |
| Network active queue management |                 aqm                 |   :heavy_check_mark:   |           :x:           |          -          |
|        CDN memory caching       |                cache                |   :heavy_check_mark:   |    :heavy_check_mark:   |         :x:         |
|          Circuit design         | circuit_three_ stage_transimpedance |   :heavy_check_mark:   |    :heavy_check_mark:   |         :x:         |
|    Network congestion control   |          congestion_control         |           :x:          |            -            |          -          |
|      Server load balancing      |             load_balance            |   :heavy_check_mark:   |    :heavy_check_mark:   |  :heavy_check_mark: |
|   Multi-dim database indexing   |           multi_dim_index           |   :heavy_check_mark:   |           :x:           |          -          |
| SQL database query optimization |           query_optimizer           |           :x:          |            -            |          -          |
|    Account region assignment    |          region_assignment          |   :heavy_check_mark:   |           :x:           |          -          |
|           Simple queue          |             simple_queue            |   :heavy_check_mark:   |    :heavy_check_mark:   |  :heavy_check_mark: |
|   Spark cluster job scheduling  |                spark                |           :x:          |            -            |          -          |
|   Spark cluster job scheduling  |              spark_sim              |   :heavy_check_mark:   |           :x:           |          -          |
|        Switch scheduling        |          switch_scheduling          |   :heavy_check_mark:   |    :heavy_check_mark:   |  :heavy_check_mark: |
|   Tensorflow device placement   |             tf_placement            |           :x:          |            -            |          -          |
|   Tensorflow device placement   |           tf_placement_sim          |   :heavy_check_mark:   |           :x:           |          -          |
|      **Process scheduling**     |        **process_scheduling**       | **:heavy_check_mark:** |  **:heavy_check_mark:** |       **:x:**       |

As we can see, the initialization succeeds for 11 out of the 17 Park environments, 6 of them can be converted to Gym environments and only 3 are compatible with RLlib. You can re-run the tests on your own device by using the `test_environments()` function described below.

## Usage
### Getting started
1. Clone this repository
   ```
   git clone https://github.com/jakobhartmann/deep-rl-system-research.git
   ```

2. Change directory
   ```
   cd deep-rl-system-research
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

### Check compatibility of Park environments with Gym wrapper and RLlib
```python
from env_wrapper import test_environments
envs = ['abr', 'abr_sim', 'aqm', 'cache', 'circuit_three_stage_transimpedance', 'congestion_control', 'load_balance', 'query_optimizer', 'multi_dim_index', 'region_assignment', 'simple_queue', 'spark', 'spark_sim', 'switch_scheduling', 'tf_placement', 'tf_placement_sim', 'process_scheduling']
env_summary = test_environments(envs = envs)
print(env_summary)
```
The function returns a dictionary in which each key corresponds to one environment. The value can take three possible forms:
1. `Success!` The initialization succeeded, the conversion to a Gym environment worked and the environment is compatible with RLlib.
2. `NotImplementedError` The initialization succeeded, but at least one Park space is not supported by Gymnasium.
3. Any other error message: The initialization of the Park environment failed.


### Run the ProcessScheduling Park environment
```python
import park
from process_scheduling_park_agents import *
env = park.make('process_scheduling')
done = False
obs, info = env.reset()
    
while not done:
    act = sjf_agent(obs) # Use Shortest-Job-First agent
    obs, reward, done, info = env.step(act)
```
Alternatively, you can also run the given example:
```
python process_scheduling_park_agents.py
```

### Register and run the ProcessScheduling Gym environment
```python
import gym
from gym.envs.registration import register
from process_scheduling_gym.process_scheduling_gym_agents import *

register(
    id='ProcessScheduling-v0',
    entry_point='process_scheduling_gym.process_scheduling_gym_env:ProcessSchedulingEnv',
    max_episode_steps=1000,
)

env = gym.make('process_scheduling_gym.process_scheduling_gym_env:ProcessScheduling-v0')
    
done = False
obs, info = env.reset()

while not done:
    act = sjf_agent(obs) # Use Shortest-Job-First agent
    obs, reward, done, info = env.step(act)
```
Alternatively, you can also run the given example:
```
python process_scheduling_gym/process_scheduling_gym_agents.py
```


### Manually check compatibility of a Park environment with RLlib
```python
from env_wrapper import ParkAgent
from ray import rllib

park_agent = ParkAgent({'name': 'process_scheduling'})
rllib.utils.check_env(park_agent)
```