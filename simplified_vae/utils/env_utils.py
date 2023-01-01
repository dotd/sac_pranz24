from typing import Union

import gym
import numpy as np
from gym import Env
import torch

from simplified_vae.config.config import Config
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.utils.vae_storage import VAEBuffer


def sample_stationary_trajectory(env: Union[Env, StationaryCheetahWindVelEnv], max_env_steps):

    # initialize env for the beginning of a new rollout
    obs = env.reset()

    # init vars
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    steps = 0
    while True:

        # use the most recent ob to decide what to do
        all_obs.append(obs)
        curr_action = env.action_space.sample()
        all_actions.append(curr_action)

        # take that action and record results
        obs, reward, done, _ = env.step(curr_action)

        # record result of taking that action
        steps += 1
        all_next_obs.append(obs)
        all_rewards.append(reward)

        # End the rollout if the rollout ended
        # Note that the rollout can end due to done, or due to max_path_length
        rollout_done = done or (steps >= max_env_steps)
        all_dones.append(done)

        if rollout_done:
            break

    return np.asarray(all_obs), \
           np.asarray(all_actions), \
           np.asarray(all_rewards)[:, np.newaxis], \
           np.asarray(all_next_obs), \
           np.asarray(all_dones)[:, np.newaxis]


def sample_non_stationary_trajectory(env: Union[Env, StationaryCheetahWindVelEnv], max_env_steps):

    # initialize env for the beginning of a new rollout
    obs = env.reset()

    change_task_idx = np.random.randint(0, max_env_steps) # TODO change to rvs()

    # init vars
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    steps = 0
    while True:

        # use the most recent ob to decide what to do
        all_obs.append(obs)
        curr_action = env.action_space.sample()
        all_actions.append(curr_action)

        # take that action and record results
        obs, reward, done, _ = env.step(curr_action)

        # record result of taking that action
        steps += 1
        all_next_obs.append(obs)
        all_rewards.append(reward)

        # End the rollout if the rollout ended
        # Note that the rollout can end due to done, or due to max_path_length
        rollout_done = done or (steps >= max_env_steps)
        all_dones.append(done)

        if change_task_idx == steps:
            env.set_task(task=None)

        if rollout_done:
            break

    return np.asarray(all_obs), \
           np.asarray(all_actions), \
           np.asarray(all_rewards)[:, np.newaxis], \
           np.asarray(all_next_obs), \
           np.asarray(all_dones)[:, np.newaxis]


def make_stationary_env(config: Config):

    # Environment
    max_episode_steps = 100

    env = gym.make(config.env_name)
    env._max_episode_steps = config.train_buffer.max_episode_len

    env = StationaryCheetahWindVelEnv(env=env, config=config)
    env._max_episode_steps = max_episode_steps

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    return env


def collect_stationary_train_trajectories(config: Config, env: Union[gym.Env, StationaryCheetahWindVelEnv], train_buffer: VAEBuffer):

    for trajectory_idx in range(config.train_buffer.max_episode_num):

        if trajectory_idx % config.training.change_env_freq == 0:
            env.set_task(task=None)
            print(f'Train: Episode idx {trajectory_idx}/{config.train_buffer.max_episode_num}, task - {env.get_task()}')

        obs, actions, rewards, next_obs, dones = sample_stationary_trajectory(env=env, max_env_steps=config.train_buffer.max_episode_len)

        train_buffer.insert(obs=obs,
                            actions=actions,
                            rewards=rewards,
                            next_obs=next_obs,
                            dones=dones)


def collect_stationary_test_trajectories(config: Config, env: Union[gym.Env, StationaryCheetahWindVelEnv], buffer: VAEBuffer):

    for trajectory_idx in range(config.test_buffer.max_episode_num):

        env.set_task(task=None)

        obs, actions, rewards, next_obs, dones = sample_stationary_trajectory(env=env, max_env_steps=config.test_buffer.max_episode_len)

        buffer.insert(obs=obs,
                      actions=actions,
                      rewards=rewards,
                      next_obs=next_obs,
                      dones=dones)


def collect_stationary_trajectories(env: Union[gym.Env,
                                    StationaryCheetahWindVelEnv],
                                    buffer: VAEBuffer,
                                    episode_num: int,
                                    episode_len: int):

    for trajectory_idx in range(episode_num):
        env.set_task(task=None)

        obs, actions, rewards, next_obs, dones = sample_stationary_trajectory(env=env, max_env_steps=episode_len)

        buffer.insert(obs=obs,
                      actions=actions,
                      rewards=rewards,
                      next_obs=next_obs,
                      dones=dones)

