import torch
import numpy as np

from simplified_vae.twr.twr_trainer import TWRTrainer
from simplified_vae.config.config import Config
from simplified_vae.utils.env_utils import make_stationary_env


def main():

    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = make_stationary_env(config=config)

    twr_trainer = TWRTrainer(config=config, env=env)

    twr_trainer.train_model()


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()