from simplified_vae.utils.env_utils import make_stationary_env
from twr.config import TWRConfig

import torch
import numpy as np

from twr.twr_trainer import TWRTrainer


def main():

    config = TWRConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = make_stationary_env(config=config)

    twr_trainer = TWRTrainer(config=config, env=env)

    twr_trainer.train_model()


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()