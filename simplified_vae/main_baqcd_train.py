import inspect
from datetime import datetime
import os
import wandb
import sys

# Add script directory to sys.path.
from torch.utils.tensorboard import SummaryWriter
from simplified_vae.config.config import BaseConfig, ModelConfig, \
    AgentConfig
from simplified_vae.config.envs_config import StationaryCheetahWindvelEnvConfig, FixedToggleCheetahWindvelEnvConfig, \
    StationaryABSEnvConfig, FixedToggleABSEnvConfig, FixedToggleHopperWindvelEnvConfig, \
    StationaryHopperWindvelEnvConfig, FixedToggleSwimmerWindvelEnvConfig, StationarySwimmerWindvelEnvConfig
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.utils.env_utils import set_seed
from simplified_vae.utils.poc_trainer import POCTrainer


def GetScriptDirectory():
    if hasattr(GetScriptDirectory, "dir"):
        return GetScriptDirectory.dir
    module_path = ""
    try:
        # The easy way. Just use __file__.
        # Unfortunately, __file__ is not available when cx_Freeze is used or in IDLE.
        module_path = __file__
    except NameError:
        if len(sys.argv) > 0 and len(sys.argv[0]) > 0 and os.path.isabs(sys.argv[0]):
            module_path = sys.argv[0]
        else:
            module_path = os.path.abspath(inspect.getfile(GetScriptDirectory))
            if not os.path.exists(module_path):
                # If cx_Freeze is used the value of the module_path variable at this point is in the following format.
                # {PathToExeFile}\{NameOfPythonSourceFile}.
                # This makes it necessary to strip off the file name to get the correct
                # path.
                module_path = os.path.dirname(module_path)
    GetScriptDirectory.dir = os.path.dirname(module_path)
    return GetScriptDirectory.dir


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
sys.path.append(os.path.join(GetScriptDirectory(), "lib"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'/simplified_vae/')))


def main():

    ### Cheetah WindVel VAE ###
    #######################
    # model_type = 'VAE'
    # checkpoint_path = 'runs/VAE_HalfCheetah-v2_2023-04-27_13-32-38/model_best.pth.tar'

    # config = BaseConfig(env=FixedToggleCheetahWindvelEnvConfig(),
    #                     model=ModelConfig(type=model_type, checkpoint_path=checkpoint_path),
    #                     agent=AgentConfig(start_steps=-1))
    #
    # stationary_config = BaseConfig(env=StationaryCheetahWindvelEnvConfig(),
    #                                model=ModelConfig(type=model_type, checkpoint_path=checkpoint_path),
    #                                agent=AgentConfig(start_steps=-1))

    ### Cheetah WindVel RNN-VAE ###
    #######################

    # model_type = 'RNNVAE'
    # checkpoint_path = 'runs/2023-01-02_09-12-57_VAE/model_best.pth.tar'
    #
    # config = BaseConfig(env=FixedToggleCheetahWindvelEnvConfig(),
    #                     model=ModelConfig(type=model_type, checkpoint_path=checkpoint_path),
    #                     agent=AgentConfig(start_steps=-1))
    #
    # stationary_config = BaseConfig(env=StationaryCheetahWindvelEnvConfig(),
    #                                model=ModelConfig(type=model_type, checkpoint_path=checkpoint_path),
    #                                agent=AgentConfig(start_steps=-1))

    ### Swimmer WindVel RNN-VAE###
    #######################

    # model_type = 'RNNVAE'
    # checkpoint_path = 'runs/RNNVAE_Swimmer-v3_2023-05-01_14-24-34/model_best.pth.tar'
    # config = BaseConfig(env=FixedToggleSwimmerWindvelEnvConfig(),
    #                     model=ModelConfig(type=model_type, checkpoint_path=checkpoint_path),
    #                     agent=AgentConfig(start_steps=-1))
    #
    # stationary_config = BaseConfig(env=StationarySwimmerWindvelEnvConfig(),
    #                                model=ModelConfig(type=model_type, checkpoint_path=checkpoint_path),
    #                                agent=AgentConfig(start_steps=-1))
    ### Hopper WindVel ###
    #######################

    config = BaseConfig(env=FixedToggleHopperWindvelEnvConfig(),
                        model=ModelConfig(checkpoint_path='runs/RNNVAE_Hopper-v3_2023-05-02_15-58-56/model_best.pth.tar'),
                        agent=AgentConfig(start_steps=1000))

    stationary_config = BaseConfig(env=StationaryHopperWindvelEnvConfig(),
                                   model=ModelConfig(checkpoint_path='runs/RNNVAE_Hopper-v3_2023-05-02_15-58-56/model_best.pth.tar'),
                                   agent=AgentConfig(start_steps=1000))

    ### ABS Env ###
    ###############

    # config = BaseConfig(env=FixedToggleABSEnvConfig(max_episode_steps=1000000),
    #                     model=ModelConfig(checkpoint_path='runs/VAE_FixedABS_2023-03-05_13-45-13/model_best.pth.tar'),
    #                     agent=AgentConfig(start_steps=-1))
    #
    # stationary_config = BaseConfig(env=StationaryABSEnvConfig(max_episode_steps=1000000),
    #                                model=ModelConfig(checkpoint_path='runs/VAE_FixedABS_2023-03-05_13-45-13/model_best.pth.tar'),
    #                                agent=AgentConfig(start_steps=-1))

    logger = SummaryWriter(f'runs/BAQCD_{config.model.type}_{config.env.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    set_seed(config.seed)

    # wandb.init(project="cusum_exps",
    #            config=config.__dict__)

    # Init Env
    env = env_factory(config=config, logger=logger)
    data_collection_env = env_factory(config=stationary_config, logger=logger)

    # init Trainer
    poc_trainer = POCTrainer(config=config,
                             env=env,
                             data_collection_env=data_collection_env,
                             logger=logger)

    poc_trainer.init_clusters()
    poc_trainer.train_model()
    # wandb.finish()

if __name__ == '__main__':
    main()
