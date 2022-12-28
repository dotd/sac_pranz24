import gym
import torch
import numpy as np
import time

from simplified_vae.config import Config
from simplified_vae.vae_trainer import VAETrainer
from vae_storage import VAEBuffer
from utils import collect_trajectories


def main():

    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = gym.make(config.env_name)
    env.seed(config.seed)

    obs_dim = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]

    vae_buffer = VAEBuffer(config=config, obs_dim=obs_dim, action_dim=action_dim)
    vae_trainer = VAETrainer(config=config, buffer=vae_buffer, obs_dim=obs_dim, action_dim=action_dim)
    start_time = time.time()

    collect_trajectories(config=config, env=env, vae_buffer=vae_buffer)

    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print(f'Took {elapsed_time} seconds to collect {config.epiosde_num} episodes')

    vae_trainer.train_model()


if __name__ == '__main__':
    main()