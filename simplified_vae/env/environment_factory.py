from typing import Optional

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig, \
    StationaryWindvelEnvConfig, \
    ToggleWindvelEnvConfig, \
    FixedToggleWindvelEnvConfig, \
    StationaryABSEnvConfig, \
    ToggleABSEnvConfig, \
    FixedToggleABSEnvConfig, StationaryHopperWindvelEnvConfig, ToggleHopperWindvelEnvConfig
from simplified_vae.env.stationary_abs_env import StationarySingleWheelEnv
from simplified_vae.env.fixed_toggle_abs_env import FixedToggleSingleWheelEnv
from simplified_vae.env.fixed_toggle_windvel_env import FixedToggleWindVelEnv
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.env.stationary_hopper_windvel import StationaryHopperWindVelEnv
from simplified_vae.env.toggle_abs_env import ToggleSingleWheelEnv
from simplified_vae.env.toggle_hopper_windvel_env import ToggleHopperWindVelEnv
from simplified_vae.env.toggle_windvel_env import ToggleWindVelEnv


def env_factory(config: BaseConfig, logger: Optional[SummaryWriter]):

    if isinstance(config.env, StationaryWindvelEnvConfig):

        env = gym.make('HalfCheetah-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = StationaryCheetahWindVelEnv(env=env, config=config, logger=logger)

    elif isinstance(config.env, ToggleWindvelEnvConfig):

        env = gym.make('HalfCheetah-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = ToggleWindVelEnv(env=env, config=config, logger=logger)

    elif isinstance(config.env, FixedToggleWindvelEnvConfig):

        env = gym.make('HalfCheetah-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = FixedToggleWindVelEnv(env=env, config=config, logger=logger)

    elif isinstance(config.env, StationaryABSEnvConfig):
        env = StationarySingleWheelEnv(config=config, logger=logger)
        env._max_episode_steps = config.env.max_episode_steps

    elif isinstance(config.env, ToggleABSEnvConfig):
        env = ToggleSingleWheelEnv(config=config, logger=logger)
        env._max_episode_steps = config.env.max_episode_steps

    elif isinstance(config.env, FixedToggleABSEnvConfig):
        env = FixedToggleSingleWheelEnv(config=config, logger=logger)
        env._max_episode_steps = config.env.max_episode_steps

    elif isinstance(config.env, StationaryHopperWindvelEnvConfig):

        env = gym.make('Hopper-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = StationaryHopperWindVelEnv(env=env, config=config, logger=logger)

    elif isinstance(config.env, ToggleHopperWindvelEnvConfig):

        env = gym.make('Hopper-v2')
        env.seed(config.seed)
        env._max_episode_steps = config.env.max_episode_steps
        env = ToggleHopperWindVelEnv(env=env, config=config, logger=logger)

    else:
        raise NotImplementedError

    env.seed(config.seed)
    env.action_space.seed(config.seed)

    return env
