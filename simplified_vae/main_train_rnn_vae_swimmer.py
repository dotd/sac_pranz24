from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.envs_config import StationarySwimmerWindvelEnvConfig
from simplified_vae.utils.env_utils import set_seed
from simplified_vae.utils.vae_trainer import VAETrainer
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.config.config import BaseConfig


def main():

    env_config = StationarySwimmerWindvelEnvConfig()
    config = BaseConfig(env=env_config)

    set_seed(seed=config.seed)

    logger = SummaryWriter(f'runs/{config.model.type}_{config.env.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    env = env_factory(config=config, logger=logger)

    vae_trainer = VAETrainer(config=config,
                             env=env,
                             logger=logger)

    vae_trainer.train_model()


if __name__ == '__main__':
    main()