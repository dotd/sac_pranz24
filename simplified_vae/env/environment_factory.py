from typing import Optional

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig
from simplified_vae.config.envs_config import StationaryCheetahWindvelEnvConfig, ToggleCheetahWindvelEnvConfig, \
    FixedToggleCheetahWindvelEnvConfig, StationaryABSEnvConfig, ToggleABSEnvConfig, FixedToggleABSEnvConfig, \
    StationaryHopperWindvelEnvConfig, ToggleHopperWindvelEnvConfig, FixedToggleHopperWindvelEnvConfig
from simplified_vae.env.fixed_toggle_hopper_windvel_wrapper import FixedToggleHopperWindVelWrapper
from simplified_vae.env.stationary_abs_env import StationaryABSEnv
from simplified_vae.env.fixed_toggle_abs_env import FixedToggleABSEnv
from simplified_vae.env.fixed_toggle_cheetah_windvel_wrapper import FixedToggleCheetahWindVelWrapper
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelWrapper
from simplified_vae.env.stationary_hopper_windvel_wrapper import StationaryHopperWindVelWrapper
from simplified_vae.env.toggle_abs_env import ToggleABSEnv
from simplified_vae.env.toggle_cheetah_windvel_wrapper import ToggleCheetahWindVelWrapper
from simplified_vae.env.toggle_hopper_windvel_wrapper import ToggleHopperWindVelWrapper


def env_factory(config: BaseConfig, logger: Optional[SummaryWriter]):

    if isinstance(config.env, StationaryCheetahWindvelEnvConfig):

        env = gym.make('HalfCheetah-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = StationaryCheetahWindVelWrapper(env=env, config=config, logger=logger)

    elif isinstance(config.env, ToggleCheetahWindvelEnvConfig):

        env = gym.make('HalfCheetah-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = ToggleCheetahWindVelWrapper(env=env, config=config, logger=logger)

    elif isinstance(config.env, FixedToggleCheetahWindvelEnvConfig):

        env = gym.make('HalfCheetah-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = FixedToggleCheetahWindVelWrapper(env=env, config=config, logger=logger)

    elif isinstance(config.env, StationaryABSEnvConfig):
        env = StationaryABSEnv(config=config, logger=logger)
        env._max_episode_steps = config.env.max_episode_steps

    elif isinstance(config.env, ToggleABSEnvConfig):
        env = ToggleABSEnv(config=config, logger=logger)
        env._max_episode_steps = config.env.max_episode_steps

    elif isinstance(config.env, FixedToggleABSEnvConfig):
        env = FixedToggleABSEnv(config=config, logger=logger)
        env._max_episode_steps = config.env.max_episode_steps

    elif isinstance(config.env, StationaryHopperWindvelEnvConfig):

        env = gym.make(id='Hopper-v3', terminate_when_unhealthy=False)
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = StationaryHopperWindVelWrapper(env=env, config=config, logger=logger)

    elif isinstance(config.env, ToggleHopperWindvelEnvConfig):

        env = gym.make('Hopper-v3', terminate_when_unhealthy=False)
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = ToggleHopperWindVelWrapper(env=env, config=config, logger=logger)

    elif isinstance(config.env, FixedToggleHopperWindvelEnvConfig):

        env = gym.make('Hopper-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = FixedToggleHopperWindVelWrapper(env=env, config=config, logger=logger)

    else:
        raise NotImplementedError

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    return env
