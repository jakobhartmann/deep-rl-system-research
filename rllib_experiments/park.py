import argparse
import ray
from ray import air, rllib, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch

from env_wrapper import ParkAgent, test_environments

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, default='load_balance', help='Environment id.')
parser.add_argument('--results_dir', type=str, default='./results/', help='Result directory.')
parser.add_argument('--framework', type=str, default='tensorflow', help='Deep learning framework.')
parser.add_argument('--num_rollout_workers', type=int, default=25, help='Number of rollout workers.')
parser.add_argument('--num_envs_per_worker', type=int, default=5, help='Number of environments per worker.')
parser.add_argument('--max_num_total_timesteps', type=int, default=1000000, help='Maximum number of total timesteps for each run.')
parser.add_argument('--episode_reward_mean_stop', type=int, default=-1, help='Episode mean reward after which a run is terminated.')
parser.add_argument('--algorithm', type=str, default='PPO', help='Episode mean reward after which a run is terminated.')


if __name__ == '__main__':
    args = parser.parse_args()

    register_env('park_agent', lambda config: ParkAgent(config))

    config = (
        PPOConfig()
        .environment(env = 'park_agent', env_config = {'name': args.env_id})
        .framework(args.framework)
        .rollouts(num_rollout_workers = args.num_rollout_workers, num_envs_per_worker = args.num_envs_per_worker)
        .training(
            lr = tune.loguniform(1e-5, 1e-1),
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
            stop = {'timesteps_total': args.max_num_total_timesteps, 'episode_reward_mean': args.episode_reward_mean_stop},
            local_dir = args.results_dir,
            callbacks = [
                WandbLoggerCallback(project = 'R244 - Open Source Project', excludes = excluded_metrics)
            ],
        ),
        param_space = config
    )

    results = tuner.fit()

    best_result = results.get_best_result()
    best_config = best_result.config

    print('Best hyperparemters for this trial: ', best_config)