# Original source: https://github.com/facebookresearch/CompilerGym/blob/stable/examples/rllib.ipynb

import argparse
import ray
from ray import air, rllib, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.rllib.agents.ppo import PPOTrainer
from itertools import islice
import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks, ConstrainedCommandline, TimeLimit


parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, default='load_balance', help='Environment id.')
parser.add_argument('--results_dir', type=str, default='./results/', help='Result directory.')
parser.add_argument('--framework', type=str, default='tensorflow', help='Deep learning framework.')
parser.add_argument('--num_rollout_workers', type=int, default=25, help='Number of rollout workers.')
parser.add_argument('--num_envs_per_worker', type=int, default=5, help='Number of environments per worker.')
parser.add_argument('--episodes_total', type=int, default=2000, help='Total number of episodes for each run.')
parser.add_argument('--episode_reward_mean_stop', type=int, default=-1, help='Episode mean reward after which a run is terminated.')
parser.add_argument('--algorithm', type=str, default='PPO', help='Episode mean reward after which a run is terminated.')


if __name__ == '__main__':
    args = parser.parse_args()

    def make_env() -> compiler_gym.envs.CompilerEnv:
        env = compiler_gym.make(
            'llvm-v0',
            observation_space = 'Autophase',
            reward_space = 'IrInstructionCountOz',
        )
        return env


    with make_env() as env:
        npb = env.datasets['npb-v0']
        chstone = env.datasets['chstone-v0']

        train_benchmarks = list(islice(npb.benchmarks(), 55))
        train_benchmarks, val_benchmarks = train_benchmarks[:50], train_benchmarks[50:]

        test_benchmarks = list(chstone.benchmarks())


    def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
        del args
        return CycleOverBenchmarks(make_env(), train_benchmarks)

    tune.register_env("compiler_gym", make_training_env)


    if ray.is_initialized():
        ray.shutdown()

    ray.init(include_dashboard = False, ignore_reinit_error = True)

    tune.register_env('compiler_gym', make_training_env)

    config = (
        PPOConfig()
        .environment(env = 'compiler_gym')
        .framework(args.framework)
        .rollouts(num_rollout_workers = args.num_rollout_workers, num_envs_per_worker = args.num_envs_per_worker)
        .training(
            lr = tune.loguniform(1e-5, 1e-2),
            clip_param = tune.choice([0.1, 0.2, 0.3]),
            grad_clip = tune.choice([20, 25, 30, 35, 40]),
            gamma = tune.uniform(0.9, 1)
        )
    )

    hyperopt_search = HyperOptSearch(
        metric = 'episode_reward_mean',
        mode = 'max'
    )

    excluded_metrics = ['agent_timesteps_total', 'episodes_this_iter', 'episodes_total', 'counters', 'info', 'iterations_since_restore', 'num_agent_steps_sampled', 'num_agent_steps_trained', 'num_env_steps_sampled', 'num_env_steps_sampled_this_iter', 'num_env_steps_trained', 'num_env_steps_trained_this_iter', 'num_faulty_episodes', 'num_healthy_workers', 'num_in_flight_async_reqs', 'num_remote_worker_restarts', 'num_steps_trained_this_iter', 'perf', 'sampler_perf', 'sampler_results',  'time_since_restore', 'time_this_iter_s', 'time_total_s', 'timers', 'timestamp', 'timesteps_since_restore', 'timesteps_total', 'training_iteration', 'warmup_time']

    tuner = tune.Tuner(
        trainable = args.algorithm,
        tune_config = tune.TuneConfig(
            search_alg = hyperopt_search
        ),
        run_config = air.RunConfig(
            stop = {
                'episodes_total': args.episodes_total
            },
            local_dir = args.results_dir,
            callbacks = [
                WandbLoggerCallback(project = 'R244 - Open Source Project', excludes = excluded_metrics)
            ],
        ),
        param_space = config
    )

    results = tuner.fit()

    agent = PPOTrainer(
        env = 'compiler_gym',
        config={
            'num_workers': 1,
            'seed': 0xCC,
            'explore': False,
        },
    )

    checkpoint = results.get_best_checkpoint(
        metric = 'episode_reward_mean', 
        mode = 'max', 
        trial = results.trials[0]
    )

    agent.restore(checkpoint)

    def run_agent_on_benchmarks(benchmarks):
        """Run agent on a list of benchmarks and return a list of cumulative rewards."""
        with make_env() as env:
            rewards = []
            for i, benchmark in enumerate(benchmarks, start=1):
                observation, done = env.reset(benchmark=benchmark), False
                while not done:
                    action = agent.compute_single_action(observation)
                    observation, _, done, _ = env.step(action)
                rewards.append(env.episode_reward)
                print(f"[{i}/{len(benchmarks)}] {env.state}")

        return rewards

    val_rewards = run_agent_on_benchmarks(val_benchmarks)
    test_rewards = run_agent_on_benchmarks(test_benchmarks)

    print('val_rewards: ', val_rewards)
    print('test_rewards: ', test_rewards)