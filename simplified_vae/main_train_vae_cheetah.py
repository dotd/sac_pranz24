from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.utils.env_utils import set_seed
from simplified_vae.utils.vae_trainer import VAETrainer
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.config.config import BaseConfig, ModelConfig, EncoderConfig
from simplified_vae.config.envs_config import StationaryCheetahWindvelEnvConfig


def main():

    env_config = StationaryCheetahWindvelEnvConfig()

    config = BaseConfig(env=env_config,
                        model=ModelConfig(type='VAE',
                                          encoder=EncoderConfig(action_embed_dim=16,
                                                                obs_embed_dim=32,
                                                                reward_embed_dim=16,
                                                                additional_embed_layers=[64, 32, 16],
                                                                recurrent_hidden_dim=128,
                                                                vae_hidden_dim=5)))

    set_seed(seed=config.seed)

    logger = SummaryWriter(f'runs/{config.model.type}_{config.env.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    env = env_factory(config=config, logger=logger)

    vae_trainer = VAETrainer(config=config,
                             env=env,
                             logger=logger)

    vae_trainer.train_model()


if __name__ == '__main__':
    main()