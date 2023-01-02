from matplotlib import pyplot as plt

import gym
import torch
import numpy as np

from simplified_vae.config.config import Config
from simplified_vae.utils.env_utils import make_stationary_env, collect_stationary_trajectories
from simplified_vae.models.vae import VAE, RNNVAE
from simplified_vae.utils.logging_utils import load_checkpoint
from simplified_vae.utils.vae_storage import VAEBuffer


def main():

    ## Init config
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Init Env
    env = make_stationary_env(config=config)
    obs_dim: int = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]
    trajectories_num = 120

    # Init model
    model: RNNVAE = RNNVAE(config=config,
                     obs_dim=obs_dim,
                     action_dim=action_dim)

    # checkpoint_path = 'runs/2023-01-01_11-55-39_VAE/model_best.pth.tar'
    checkpoint_path = 'runs/2023-01-02_09-12-57_VAE/model_best.pth.tar'

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
                                    env_change_freq=1)

    # collect episode from Task_1
    env.set_task(task=None)
    task_1 = env.get_task()

    collect_stationary_trajectories(env=env,
                                    buffer=test_buffer,
                                    episode_num=trajectories_num,
                                    episode_len=config.train_buffer.max_episode_len,
                                    env_change_freq=1)

    # Get obs encodings
    model.eval()
    with torch.no_grad():

        obs_0, actions_0, rewards_0, next_obs_0 = test_buffer.sample_section(start_idx=0, end_idx=trajectories_num)
        obs_1, actions_1, rewards_1, next_obs_1 = test_buffer.sample_section(start_idx=trajectories_num, end_idx=trajectories_num*2)

        # non-stationary trajectory
        obs_2      = torch.cat([obs_0[:, :config.train_buffer.max_episode_len // 2, :], obs_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)
        actions_2  = torch.cat([actions_0[:, :config.train_buffer.max_episode_len // 2, :], actions_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)
        rewards_2  = torch.cat([rewards_0[:, :config.train_buffer.max_episode_len // 2, :], rewards_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)
        next_obs_2 = torch.cat([next_obs_0[:, :config.train_buffer.max_episode_len // 2, :], next_obs_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        # Reverse trajectory
        obs_3      = torch.cat([obs_0[:, config.train_buffer.max_episode_len // 2:, :], obs_0[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)
        actions_3  = torch.cat([actions_0[:, config.train_buffer.max_episode_len // 2:, :], actions_0[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)
        rewards_3  = torch.cat([rewards_0[:, config.train_buffer.max_episode_len // 2:, :], rewards_0[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)
        next_obs_3 = torch.cat([next_obs_0[:, config.train_buffer.max_episode_len // 2:, :], next_obs_0[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        obs_0_d = obs_0.to(config.device)
        obs_1_d = obs_1.to(config.device)
        obs_2_d = obs_2.to(config.device)
        obs_3_d = obs_3.to(config.device)

        actions_0_d = actions_0.to(config.device)
        actions_1_d = actions_1.to(config.device)
        actions_2_d = actions_2.to(config.device)
        actions_3_d = actions_3.to(config.device)

        rewards_0_d = rewards_0.to(config.device)
        rewards_1_d = rewards_1.to(config.device)
        rewards_2_d = rewards_2.to(config.device)
        rewards_3_d = rewards_3.to(config.device)

        next_obs_0_d = next_obs_0.to(config.device)
        next_obs_1_d = next_obs_1.to(config.device)
        next_obs_2_d = next_obs_2.to(config.device)
        next_obs_3_d = next_obs_3.to(config.device)

        latent_sample_0, latent_mean_0, latent_logvar_0, output_0 = model.encoder(obs=obs_0_d, actions=actions_0_d, rewards=rewards_0_d)
        latent_sample_1, latent_mean_1, latent_logvar_1, output_1 = model.encoder(obs=obs_1_d, actions=actions_1_d, rewards=rewards_1_d)
        latent_sample_2, latent_mean_2, latent_logvar_2, output_2 = model.encoder(obs=obs_2_d, actions=actions_2_d, rewards=rewards_2_d)
        latent_sample_3, latent_mean_3, latent_logvar_3, output_3 = model.encoder(obs=obs_3_d, actions=actions_3_d, rewards=rewards_3_d)

    latent_mean_0 = latent_mean_0.cpu().numpy()
    latent_mean_1 = latent_mean_1.cpu().numpy()
    latent_mean_2 = latent_mean_2.cpu().numpy()
    latent_mean_3 = latent_mean_3.cpu().numpy()

    latent_logvar_0 = latent_logvar_0.cpu().numpy()
    latent_logvar_1 = latent_logvar_1.cpu().numpy()
    latent_logvar_2 = latent_logvar_2.cpu().numpy()
    latent_logvar_3 = latent_logvar_3.cpu().numpy()

    episode_idx = 30
    a_0, a_00 = latent_mean_0[episode_idx, :, 0], latent_logvar_0[episode_idx, :, 0]
    b_0, b_00 = latent_mean_0[episode_idx, :, 1], latent_logvar_0[episode_idx, :, 1]
    c_0, c_00 = latent_mean_0[episode_idx, :, 2], latent_logvar_0[episode_idx, :, 2]
    d_0, d_00 = latent_mean_0[episode_idx, :, 3], latent_logvar_0[episode_idx, :, 3]
    e_0, e_00 = latent_mean_0[episode_idx, :, 4], latent_logvar_0[episode_idx, :, 4]

    a_1, a_11 = latent_mean_1[episode_idx, :, 0], latent_logvar_1[episode_idx, :, 0]
    b_1, b_11 = latent_mean_1[episode_idx, :, 1], latent_logvar_1[episode_idx, :, 1]
    c_1, c_11 = latent_mean_1[episode_idx, :, 2], latent_logvar_1[episode_idx, :, 2]
    d_1, d_11 = latent_mean_1[episode_idx, :, 3], latent_logvar_1[episode_idx, :, 3]
    e_1, e_11 = latent_mean_1[episode_idx, :, 4], latent_logvar_1[episode_idx, :, 4]

    a_2, a_22 = latent_mean_2[episode_idx, :, 0], latent_logvar_2[episode_idx, :, 0]
    b_2, b_22 = latent_mean_2[episode_idx, :, 1], latent_logvar_2[episode_idx, :, 1]
    c_2, c_22 = latent_mean_2[episode_idx, :, 2], latent_logvar_2[episode_idx, :, 2]
    d_2, d_22 = latent_mean_2[episode_idx, :, 3], latent_logvar_2[episode_idx, :, 3]
    e_2, e_22 = latent_mean_2[episode_idx, :, 4], latent_logvar_2[episode_idx, :, 4]

    a_3, a_33 = latent_mean_3[episode_idx, :, 0], latent_logvar_3[episode_idx, :, 0]
    b_3, b_33 = latent_mean_3[episode_idx, :, 1], latent_logvar_3[episode_idx, :, 1]
    c_3, c_33 = latent_mean_3[episode_idx, :, 2], latent_logvar_3[episode_idx, :, 2]
    d_3, d_33 = latent_mean_3[episode_idx, :, 3], latent_logvar_3[episode_idx, :, 3]
    e_3, e_33 = latent_mean_3[episode_idx, :, 4], latent_logvar_3[episode_idx, :, 4]

    plt.figure()
    plt.plot(a_0, color='red')
    plt.plot(b_0, color='green')
    plt.plot(c_0, color='blue')
    plt.plot(d_0, color='purple')
    plt.plot(e_0, color='orange')
    plt.show(block=True)

    plt.figure()
    plt.plot(a_1, color='red')
    plt.plot(b_1, color='green')
    plt.plot(c_1, color='blue')
    plt.plot(d_1, color='purple')
    plt.plot(e_1, color='orange')
    plt.show(block=True)

    plt.figure()
    plt.plot(a_0, color='red')
    plt.plot(a_2, color='blue')
    plt.show(block=True)

    plt.figure()
    plt.plot(a_0, color='red')
    plt.plot(a_1, color='green')
    plt.plot(a_2, color='blue')
    plt.plot(a_3, color='purple')
    plt.show(block=True)

    plt.figure()
    plt.plot(b_0, color='red')
    plt.plot(b_1, color='green')
    plt.plot(b_2, color='blue')
    plt.plot(b_3, color='purple')
    plt.show(block=True)

    plt.figure()
    plt.plot(c_0, color='red')
    plt.plot(c_1, color='green')
    plt.plot(c_2, color='blue')
    plt.plot(c_3, color='purple')
    plt.show(block=True)




if __name__ == '__main__':
    main()
