from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig
from simplified_vae.config.envs_config import StationaryCheetahWindvelEnvConfig, ToggleCheetahWindvelEnvConfig
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.utils.env_utils import set_seed
from simplified_vae.utils.poc_trainer import POCTrainer


def main():

    ## Init config
    config = BaseConfig(env=ToggleCheetahWindvelEnvConfig())
    stationary_config = BaseConfig(env=StationaryCheetahWindvelEnvConfig())

    set_seed(config.seed)
    logger = SummaryWriter(f'runs/BAQCD_{config.env.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    # Init Env
    env = env_factory(config=config, logger=logger)
    data_collection_env = env_factory(config=stationary_config)

    # init Trainer
    poc_trainer = POCTrainer(config=config,
                             env=env,
                             data_collection_env=data_collection_env,
                             logger=logger)

    poc_trainer.init_clusters()

    poc_trainer.test_model()


if __name__ == '__main__':
    main()