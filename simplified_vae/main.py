import gym
import torch
import numpy as np

from simplified_vae.config.config import Config
from simplified_vae.utils.vae_trainer import VAETrainer


def main():

    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = gym.make(config.env_name)
    env.seed(config.seed)

    vae_trainer = VAETrainer(config=config, env=env)

    vae_trainer.train_model()


if __name__ == '__main__':
    main()