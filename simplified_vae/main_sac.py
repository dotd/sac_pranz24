import os
import wandb
import sys

# from simplified_vae.config.config import Config
# from simplified_vae.utils.env_utils import make_stationary_env, set_seed, make_toggle_env, make_fixed_toggle_env
# from simplified_vae.utils.poc_trainer import POCTrainer

# Add script directory to sys.path.
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
# print(os.getcwd())

from simplified_vae.config.config import Config
from simplified_vae.utils.env_utils import make_stationary_env, set_seed, make_toggle_env, make_fixed_toggle_env
from simplified_vae.utils.poc_trainer import POCTrainer

def main():

    ## Init config
    config = Config()

    wandb.init(
        # set the wandb project where this run will be logged
        project="cusum_exps",

        # track hyperparameters and run metadata
        config=config.__dict__
    )

    set_seed(config.seed)

    # Init Env
    env = make_fixed_toggle_env(config=config)
    data_collection_env = make_stationary_env(config=config)

    # init Trainer
    poc_trainer = POCTrainer(config=config,
                             env=env,
                             data_collection_env=data_collection_env)

    poc_trainer.init_clusters()

    poc_trainer.train_model()
    wandb.finish()

if __name__ == '__main__':
    main()
