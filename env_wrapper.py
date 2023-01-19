# Source: https://github.com/park-project/park/blob/b533ba368881e5424fbe22280b8a86eb5a3e613c/algorithms/agent_wrapper.py

import park
import gym 
from ray import rllib

class ParkAgent(gym.Env):
    def __init__(self, env_config = None):
        self.env = park.make(env_config['name'])
        self.action_space = self.get_gym_space(self.env.action_space)
        self.observation_space = self.get_gym_space(self.env.observation_space)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def get_gym_space(self, space):
        if type(space) == park.spaces.Box:
            return self.get_gym_box_space(space)
        elif type(space) == park.spaces.Tuple:
            return self.get_gym_tuple_space(space)
        elif type(space) == park.spaces.Graph:
            return self.get_gym_graph_space(space)
        elif type(space) == park.spaces.Discrete:
            return self.get_gym_discrete_space(space)
        elif type(space) == park.spaces.Dict:
            return self.get_gym_dict_space(space)

        print('Space type: ', type(space))
        raise NotImplementedError

    def get_gym_box_space(self, space): 
        return gym.spaces.Box(
            low = space.low,
            high = space.high,
            shape = space.shape, 
            dtype = space.dtype
        )

    def get_gym_discrete_space(self, space):
        return gym.spaces.Discrete(space.n)

    def get_gym_dict_space(self, space):
        spaces = {}
        for key, value in space.spaces.items():
            spaces[key] = self.get_gym_space(value)

        return gym.spaces.Dict(
            spaces = spaces
        )
    
    def get_gym_tuple_space(self, space):
        spaces = ()
        for space in space.spaces:
            new = (*spaces, self.get_gym_space(space))
        return gym.spaces.Tuple(
            spaces = spaces
        )

    def get_gym_graph_space(self, space):
        return gym.spaces.Graph(
            node_space = self.get_gym_space(space.node_feature_space),
            edge_space = self.get_gym_space(space.edge_feature_space)
        )


class ParkDiscreteSpace(gym.spaces.Discrete):
    def __init__(self, discrete_space):
        self.park_discrete_space = discrete_space
        self.n = discrete_space.n

    def sample(self):
        return self.park_discrete_space.sample()

    def contains(self, x):
        return self.park_discrete_space.contains(x)

class ParkBoxSpace(gym.spaces.Box):
    def __init__(self, box_space):
        self.park_box_space = box_space
        self.shape = box_space.shape
        self.high = box_space.high
        self.low = box_space.low
        self.struct = box_space.struct
        self.dtype = box_space.dtype

    def sample(self):
        return self.park_box_space.sample()

    def contains(self, x):
        return self.park_box_space.contains(x)


def test_environments(envs = ['abr', 'abr_sim', 'aqm', 'cache', 'circuit_three_stage_transimpedance', 'congestion_control', 'load_balance', 'query_optimizer', 'multi_dim_index', 'region_assignment', 'simple_queue', 'spark', 'spark_sim', 'switch_scheduling', 'tf_placement', 'tf_placement_sim', 'process_scheduling']):
    env_summary = {}
    for env_name in envs:
        try:
            env = park.make(env_name)
            park_agent = ParkAgent({'name': env_name})
            rllib.utils.check_env(park_agent)
            env_summary[env_name] = 'Success!'
        except Exception as e:
            env_summary[env_name] = e
            continue
    return env_summary