import inspect
from datetime import datetime
import os
import wandb
import sys

# Add script directory to sys.path.
from torch.utils.tensorboard import SummaryWriter
from simplified_vae.config.config import BaseConfig, ToggleWindvelEnvConfig, StationaryWindvelEnvConfig
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

    config = BaseConfig(env=ToggleWindvelEnvConfig())
    stationary_config = BaseConfig(env=StationaryWindvelEnvConfig())

    logger = SummaryWriter(f'runs/BAQCD_{config.env.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    set_seed(config.seed)

    wandb.init(project="cusum_exps",
               config=config.__dict__)

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
    wandb.finish()

if __name__ == '__main__':
    main()
