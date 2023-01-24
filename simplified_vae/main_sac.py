from simplified_vae.config.config import Config
from simplified_vae.utils.env_utils import make_stationary_env, set_seed, make_toggle_env, make_fixed_toggle_env
from simplified_vae.utils.poc_trainer import POCTrainer
import wandb

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