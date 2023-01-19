import numpy as np
import gym
from gym import spaces

class ProcessSchedulingEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(self, seed = 42):
        self.seed = seed
        self.cpu1 = np.array([0])
        self.processes = np.random.randint(low = 1, high = 10, size = 20)

        self.observation_space = spaces.Dict(
            {
                'cpu1': spaces.Discrete(1),
                'processes': spaces.Discrete(len(self.processes))
            }
        )

        self.action_space = spaces.Discrete(len(self.processes))


    def step(self, action):
        # If the CPU is not idle, reduce the duration of the current process by 1
        if self.cpu1[0] > 0:
            self.cpu1[0] -= 1

        # Only apply action, if CPU is idle
        if self.cpu1[0] == 0:
            # Assign process to CPU
            self.cpu1[0] = self.processes[action]
            self.processes[action] = 0

        observation = self._get_obs()
        info = self._get_info()

        reward = -len(list(filter(lambda x: x > 0, self.processes)))
        if reward == 0 and self.cpu1[0] == 0:
            done = True
        else:
            done = False

        return observation, reward, done, info


    def reset(self, options = None):
        self.__init__()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def _get_obs(self):
        return {'CPU1': self.cpu1, 'processes': self.processes}


    def _get_info(self):
        return {'CPU1': self.cpu1, 'processes': self.processes}


    def close(self):
        pass