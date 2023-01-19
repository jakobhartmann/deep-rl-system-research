import gym
from gym.envs.registration import register


# First-In First-Out
def fifo_agent(observation):
    processes = observation['processes']
    action = 0
    for i in range(len(processes)):
        if 0 < processes[i]:
            action = i
            break
    return action


# First-In Last-Out
def filo_agent(observation):
    processes = observation['processes']
    action = 0
    for i in range(len(processes) - 1, 0, -1):
        if 0 < processes[i]:
            action = i
            break
    return action


# Shortest-Job-First
def sjf_agent(observation):
    processes = observation['processes']
    action = float('inf')
    for i in range(len(processes)):
        if 0 < processes[i] < action:
            action = i
    if action == float('inf'):
        action = 0
    return action



if __name__ == '__main__':
    register(
        id='ProcessScheduling-v0',
        entry_point='process_scheduling_gym_env:ProcessSchedulingEnv',
        max_episode_steps=1000,
    )

    env = gym.make('process_scheduling_gym_env:ProcessScheduling-v0')
    
    iteration = 0
    done = False
    cumulative_reward = 0
    obs, info = env.reset()
    while not done:
        iteration += 1
        # act = env.action_space.sample() # random agent
        act = sjf_agent(obs)
        obs, reward, done, info = env.step(act)
        cumulative_reward += reward

        if done:
            print('iteration: ', iteration)
            print('act: ', act)
            print('obs: ', obs)
            print('reward: ', reward)
            print('cumulative_reward: ', cumulative_reward)
            print('done: ', done)
            print('info: ', info)