import random
from typing import Union, List, Optional

import gym
import numpy as np
from gym import Env
import torch
import torch.nn as nn
from scipy.stats import randint

from simplified_vae.config.config import BaseConfig
from simplified_vae.env.fixed_toggle_windvel_env import FixedToggleWindVelEnv
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.env.toggle_windvel_env import ToggleWindVelEnv
from simplified_vae.models.sac import SAC
from simplified_vae.utils.vae_storage import Buffer


def sample_stationary_trajectory(env: Union[Env, StationaryCheetahWindVelEnv],
                                 max_env_steps: int,
                                 agent: Optional[SAC] = None):

    # initialize env for the beginning of a new rollout
    obs = env.reset()

    # init vars
    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    steps = 0
    while True:

        # use the most recent ob to decide what to do
        all_obs.append(obs)
        if agent:
            curr_action = agent.select_action(obs)
        else:
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


def sample_non_stationary_trajectory(env: Union[Env, StationaryCheetahWindVelEnv], max_env_steps: int, rg: object) -> object:

    # initialize env for the beginning of a new rollout
    obs = env.reset()

    change_task_idx_rvs = randint(0, max_env_steps)
    change_task_idx = change_task_idx_rvs.rvs(random_state=rg)

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


def make_stationary_env(config: BaseConfig):

    # Environment
    max_episode_steps = 100

    env = gym.make(config.env.name)
    env.seed(config.seed)
    env._max_episode_steps = config.train_buffer.max_episode_len

    env = StationaryCheetahWindVelEnv(env=env, config=config)
    env._max_episode_steps = max_episode_steps

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    return env


def make_toggle_env(config: BaseConfig):

    # Environment
    max_episode_steps = 100

    env = gym.make(config.env.name)
    env.seed(config.seed)
    env._max_episode_steps = config.train_buffer.max_episode_len

    env = ToggleWindVelEnv(env=env, config=config)
    env._max_episode_steps = max_episode_steps

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env.set_task(env.tasks[0])

    return env


def make_fixed_toggle_env(config: BaseConfig):

    # Environment
    max_episode_steps = 100 # TODO fix

    env = gym.make(config.env.name)
    env.seed(config.seed)
    env._max_episode_steps = config.train_buffer.max_episode_len

    env = FixedToggleWindVelEnv(env=env, config=config)
    env._max_episode_steps = max_episode_steps

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env.set_task(env.tasks[0])

    return env


def collect_stationary_trajectories(env: Union[gym.Env,
                                    StationaryCheetahWindVelEnv],
                                    buffer: Buffer,
                                    episode_num: int,
                                    episode_len: int,
                                    env_change_freq: int,
                                    agent: Optional[SAC] = None):

    for trajectory_idx in range(episode_num):

        if trajectory_idx % env_change_freq == 0 and trajectory_idx > 0:
            env.set_task(task=None)
            curr_task = env.get_task()
            print(f'Task Changed to {curr_task}')

        if trajectory_idx % 100 == 0:
            print(f'Train: Episode idx {trajectory_idx}/{episode_num}, task - {env.get_task()}')

        obs, actions, rewards, next_obs, dones = sample_stationary_trajectory(env=env,
                                                                              max_env_steps=episode_len,
                                                                              agent=agent)

        buffer.insert(obs=obs,
                      actions=actions,
                      rewards=rewards,
                      next_obs=next_obs,
                      dones=dones)


def collect_non_stationary_trajectories(env: Union[gym.Env, StationaryCheetahWindVelEnv],
                                        buffer: Buffer,
                                        episode_num: int,
                                        episode_len: int,
                                        rg: np.random.RandomState):

    for trajectory_idx in range(episode_num):

        if trajectory_idx % 100 == 0 and trajectory_idx > 0:
            print(f'Train: Episode idx {trajectory_idx}/{episode_num}')

        obs, actions, rewards, next_obs, dones = sample_non_stationary_trajectory(env=env,
                                                                                  max_env_steps=episode_len,
                                                                                  rg=rg)

        buffer.insert(obs=obs,
                      actions=actions,
                      rewards=rewards,
                      next_obs=next_obs,
                      dones=dones)


def collect_toggle_trajectories(env: Union[gym.Env, StationaryCheetahWindVelEnv],
                                buffer: Buffer,
                                episode_num: int,
                                episode_len: int,
                                tasks: List[np.ndarray],
                                agent: Optional[SAC] = None):

    task_num = len(tasks)
    per_task_episode_num = episode_num // task_num
    task_idx = 0

    for trajectory_idx in range(episode_num):

        if trajectory_idx % per_task_episode_num == 0:
            env.set_task(tasks[task_idx])
            print(f'Set Task {tasks[task_idx]} in idx {trajectory_idx}/{episode_num}')
            task_idx += 1

        if trajectory_idx % 100 == 0 and trajectory_idx > 0:
            print(f'Train: Episode idx {trajectory_idx}/{episode_num}')

        obs, actions, rewards, next_obs, dones = sample_stationary_trajectory(env=env,
                                                                              max_env_steps=episode_len,
                                                                              agent=agent)

        buffer.insert(obs=obs,
                      actions=actions,
                      rewards=rewards,
                      next_obs=next_obs,
                      dones=dones)


def set_seed(seed: int):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
