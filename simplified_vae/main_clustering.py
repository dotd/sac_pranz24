import torch
import numpy as np

from simplified_vae.clustering.cluter_utils import latent_clustering, create_transition_matrix
from simplified_vae.config.config import Config
from simplified_vae.models.vae import RNNVAE
from simplified_vae.utils.env_utils import make_stationary_env, collect_stationary_trajectories
from simplified_vae.utils.logging_utils import load_checkpoint
from simplified_vae.utils.vae_storage import VAEBuffer


def main():

    ## Init config
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rg = np.random.RandomState(seed=config.seed)

    # Init Env
    env = make_stationary_env(config=config)
    obs_dim: int = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]
    trajectories_num = 1000

    # Init model
    model: RNNVAE = RNNVAE(config=config,
                           obs_dim=obs_dim,
                           action_dim=action_dim)

    # checkpoint_path = 'runs/2023-01-01_11-55-39_VAE/model_best.pth.tar'
    checkpoint_path = 'runs/2023-01-02_09-12-57_VAE/model_best.pth.tar'
    # checkpoint_path = 'runs/2023-01-02_11-54-10_VAE/model_best.pth.tar'

    model, epoch, loss = load_checkpoint(checkpoint_path=checkpoint_path, model=model, optimizer=None)

    # Init Buffer
    test_buffer = VAEBuffer(config=config.train_buffer, obs_dim=obs_dim, action_dim=action_dim)

    # Collect episodes from Task_0
    env.set_task(task=None)
    task_0 = env.get_task()
    collect_stationary_trajectories(env=env,
                                    buffer=test_buffer,
                                    episode_num=trajectories_num,
                                    episode_len=config.train_buffer.max_episode_len,
                                    env_change_freq=trajectories_num)

    # collect episode from Task_1
    env.set_task(task=None)
    task_1 = env.get_task()

    collect_stationary_trajectories(env=env,
                                    buffer=test_buffer,
                                    episode_num=trajectories_num,
                                    episode_len=config.train_buffer.max_episode_len,
                                    env_change_freq=trajectories_num)

    model.eval()
    with torch.no_grad():
        obs_0, actions_0, rewards_0, next_obs_0 = test_buffer.sample_section(start_idx=0, end_idx=trajectories_num)
        obs_1, actions_1, rewards_1, next_obs_1 = test_buffer.sample_section(start_idx=trajectories_num,
                                                                             end_idx=trajectories_num * 2)

        obs_0_d = obs_0.to(config.device)
        obs_1_d = obs_1.to(config.device)

        actions_0_d = actions_0.to(config.device)
        actions_1_d = actions_1.to(config.device)

        rewards_0_d = rewards_0.to(config.device)
        rewards_1_d = rewards_1.to(config.device)

        next_obs_0_d = next_obs_0.to(config.device)
        next_obs_1_d = next_obs_1.to(config.device)

        latent_sample_0, latent_mean_0, latent_logvar_0, output_0 = model.encoder(obs=obs_0_d, actions=actions_0_d, rewards=rewards_0_d)
        latent_sample_1, latent_mean_1, latent_logvar_1, output_1 = model.encoder(obs=obs_1_d, actions=actions_1_d, rewards=rewards_1_d)

        latent_mean = torch.cat([latent_mean_0, latent_mean_0], dim=0)

        latent_mean_h = latent_mean.detach.cpu().numpy()

        kmeans = latent_clustering(latent_mean_h, None, config.clustering.clusters_num, rg)

        transition_mat = create_transition_matrix(config, kmeans, config.clustering.clusters_num)






if '__name__' == '__main__':
    main()