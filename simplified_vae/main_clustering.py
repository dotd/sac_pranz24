import numpy as np
import torch
import gym

from simplified_vae.clustering.cluter_utils import latent_clustering
from simplified_vae.config.config import Config
from simplified_vae.cusum.cusum_utils import MarkovDistribution, run_cusum
from simplified_vae.utils.env_utils import make_stationary_env, collect_stationary_trajectories, set_seed
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import VAEBuffer


def main():

    ## Init config
    config = Config()
    rg = set_seed(config.seed)

    # Init Env
    env = make_stationary_env(config=config)
    obs_dim: int = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]
    trajectories_num = 1000

    # Init model
    # checkpoint_path = 'runs/2023-01-01_11-55-39_VAE/model_best.pth.tar'
    # checkpoint_path = 'runs/2023-01-02_11-54-10_VAE/model_best.pth.tar'
    checkpoint_path = 'runs/2023-01-02_09-12-57_VAE/model_best.pth.tar'

    model, epoch, loss = init_model(model_type='RNNVAE',
                                    checkpoint_path=checkpoint_path,
                                    config=config,
                                    obs_dim=obs_dim,
                                    action_dim=action_dim)

    # Init Buffer
    test_buffer = VAEBuffer(config=config.test_buffer, obs_dim=obs_dim, action_dim=action_dim)

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
        obs_0, actions_0, rewards_0, next_obs_0 = test_buffer.sample_section(start_idx=0,
                                                                             end_idx=trajectories_num)
        obs_1, actions_1, rewards_1, next_obs_1 = test_buffer.sample_section(start_idx=trajectories_num,
                                                                             end_idx=trajectories_num * 2)

        obs_0_d, actions_0_d, rewards_0_d, next_obs_0_d = all_to_device(obs_0, actions_0, rewards_0, next_obs_0, device=config.device)
        obs_1_d, actions_1_d, rewards_1_d, next_obs_1_d = all_to_device(obs_1, actions_1, rewards_1, next_obs_1, device=config.device)

        latent_sample_0, latent_mean_0, latent_logvar_0, output_0 = model.encoder(obs=obs_0_d, actions=actions_0_d, rewards=rewards_0_d)
        latent_sample_1, latent_mean_1, latent_logvar_1, output_1 = model.encoder(obs=obs_1_d, actions=actions_1_d, rewards=rewards_1_d)

        latent_mean = torch.cat([latent_mean_0, latent_mean_0], dim=0)

        latent_mean_h = latent_mean.detach().cpu().numpy()

        kmeans = latent_clustering(latent_mean_h, config.clustering.clusters_num, rg)

        markov_dist_0 = MarkovDistribution(state_num=config.clustering.clusters_num, clustering=kmeans)
        markov_dist_1 = MarkovDistribution(state_num=config.clustering.clusters_num, clustering=kmeans)

        markov_dist_0.update_transitions(trajectories=latent_mean_0)
        markov_dist_1.update_transitions(trajectories=latent_mean_1)

        test_buffer.clear()
        # Collect episodes from Task_0
        env.set_task(task=task_0)
        collect_stationary_trajectories(env=env,
                                        buffer=test_buffer,
                                        episode_num=trajectories_num,
                                        episode_len=config.train_buffer.max_episode_len,
                                        env_change_freq=trajectories_num)

        # collect episode from Task_1
        env.set_task(task=task_1)
        collect_stationary_trajectories(env=env,
                                        buffer=test_buffer,
                                        episode_num=trajectories_num,
                                        episode_len=config.train_buffer.max_episode_len,
                                        env_change_freq=trajectories_num)

        obs_0, actions_0, rewards_0, next_obs_0 = test_buffer.sample_section(start_idx=0,
                                                                             end_idx=trajectories_num)
        obs_1, actions_1, rewards_1, next_obs_1 = test_buffer.sample_section(start_idx=trajectories_num,
                                                                             end_idx=trajectories_num * 2)

        obs_2 = torch.cat([obs_0[:, :config.train_buffer.max_episode_len // 2, :],
                           obs_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        actions_2 = torch.cat([actions_0[:, :config.train_buffer.max_episode_len // 2, :],
                               actions_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        rewards_2 = torch.cat([rewards_0[:, :config.train_buffer.max_episode_len // 2, :],
                               rewards_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        next_obs_2 = torch.cat([next_obs_0[:, :config.train_buffer.max_episode_len // 2, :],
                                next_obs_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        obs_2_d, actions_2_d, rewards_2_d, next_obs_2_d = all_to_device(obs_2, actions_2, rewards_2, next_obs_2, device=config.device)
        latent_sample_2, latent_mean_2, latent_logvar_2, output_0 = model.encoder(obs=obs_2_d, actions=actions_2_d, rewards=rewards_2_d)

        sample_joint_trajectory = latent_mean_2[0, ...]
        sample_joint_trajectory = sample_joint_trajectory.detach().cpu().numpy()

        curr_labels = kmeans.predict(sample_joint_trajectory)
        curr_transitions = np.stack([curr_labels[:-1], curr_labels[1:]], axis=1)
        n_c, g_k = run_cusum(samples=curr_transitions, distribution_0=markov_dist_0, distribution_1=markov_dist_1, threshold=10)



if __name__ == '__main__':
    main()