import gym
import torch
import numpy as np

from simplified_vae.config.config import Config
from simplified_vae.env.env_utils import make_env
from simplified_vae.utils.vae_trainer import VAETrainer


def main():

    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = make_env(config=config)

    vae_trainer = VAETrainer(config=config, env=env)

    vae_trainer.train_model()


if __name__ == '__main__':
    main()