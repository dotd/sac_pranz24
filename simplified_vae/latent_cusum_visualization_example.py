import math

import numpy as np
import torch
import gym
from matplotlib import pyplot as plt

from simplified_vae.clustering.cluter_utils import latent_clustering
from simplified_vae.config.config import BaseConfig, StationaryWindvelEnvConfig
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.utils.markov_dist import MarkovDistribution
from simplified_vae.utils.env_utils import collect_stationary_trajectories, set_seed
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import Buffer


def run_cusum(curr_transitions, markov_dist_0, markov_dist_1):

    n_c, s_k, S_k, g_k = 0, [], [], []
    sample_len = len(curr_transitions)
    done = False

    for k in range(sample_len):

        curr_sample = curr_transitions[k, :]

        p_0 = markov_dist_0.pdf(curr_sample)
        p_1 = markov_dist_1.pdf(curr_sample)

        s_k.append(math.log(p_1 / p_0))
        S_k.append(sum(s_k))

        min_S_k = min(S_k)
        g_k.append(S_k[-1] - min_S_k)

        if g_k[-1] > 15 and not done:
            n_c = k #S_k.index(min(S_k))
            done = True
            #break

    print(f'n_c = {n_c}')
    plt.figure(), plt.plot(g_k), plt.show(block=True)
    return n_c, g_k

def main():

    ## Init config

    checkpoint_path = 'runs/2023-01-02_09-12-57_VAE/model_best.pth.tar' # Our approach
    # checkpoint_path = 'runs/2023-01-23_15-54-51_VAE/model_best.pth.tar' # VARIBAD with non-stationary trajectories
    # checkpoint_path = 'runs/2023-01-24_09-06-11_VAE/model_best.pth.tar' # VARIBAD with stationary trajectories

    config = BaseConfig(env=StationaryWindvelEnvConfig())  # StationaryWindvelEnvConfig or StationaryABSEnvConfig
    config.model.checkpoint_path = checkpoint_path
    rg = set_seed(config.seed)

    # Init Env
    env = env_factory(config=config, logger=None)

    obs_dim: int = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

    episode_num = 200
    max_episode_len = 100
    clusters_num = 10

    model, epoch, loss = init_model(config=config,
                                    obs_dim=obs_dim,
                                    action_dim=action_dim)

    # Init Buffer
    test_buffer = Buffer(max_episode_num=episode_num,
                         max_episode_len=max_episode_len,
                         obs_dim=obs_dim, action_dim=action_dim)

    # Collect episodes from Task_0
    env.set_task(task=None)
    task_0 = env.get_task()
    collect_stationary_trajectories(env=env,
                                    buffer=test_buffer,
                                    episode_num=episode_num // 2,
                                    episode_len=max_episode_len,
                                    env_change_freq=episode_num)

    # collect episode from Task_1
    env.set_task(task=None)
    task_1 = env.get_task()
    collect_stationary_trajectories(env=env,
                                    buffer=test_buffer,
                                    episode_num=episode_num // 2,
                                    episode_len=max_episode_len,
                                    env_change_freq=episode_num)

    model.eval()
    with torch.no_grad():
        obs_0, actions_0, rewards_0, next_obs_0 = test_buffer.sample_section(start_idx=0,
                                                                             end_idx=episode_num // 2)
        obs_1, actions_1, rewards_1, next_obs_1 = test_buffer.sample_section(start_idx=episode_num // 2,
                                                                             end_idx=episode_num)

        obs_0_d, actions_0_d, rewards_0_d, next_obs_0_d = all_to_device(obs_0, actions_0, rewards_0, next_obs_0, device=config.device)
        obs_1_d, actions_1_d, rewards_1_d, next_obs_1_d = all_to_device(obs_1, actions_1, rewards_1, next_obs_1, device=config.device)

        latent_sample_0, latent_mean_0, latent_logvar_0, output_0, hidden_state = model.encoder(obs=obs_0_d, actions=actions_0_d, rewards=rewards_0_d)
        latent_sample_1, latent_mean_1, latent_logvar_1, output_1, hidden_state = model.encoder(obs=obs_1_d, actions=actions_1_d, rewards=rewards_1_d)

        latent_mean = torch.cat([latent_mean_0, latent_mean_1], dim=0)

        latent_mean_h = latent_mean.detach().cpu().numpy()

        kmeans = latent_clustering(latent_mean_h, clusters_num, rg)

        markov_dist_0 = MarkovDistribution(state_num=clusters_num,
                                           window_length=max_episode_len,
                                           clustering=kmeans)
        markov_dist_1 = MarkovDistribution(state_num=clusters_num,
                                           window_length=max_episode_len,
                                           clustering=kmeans)

        batch_size, seq_len, latent_dim = latent_mean_h.shape

        # reshape to (-1, latent_dim) --> size will be samples X latent_dim
        data = latent_mean_h.reshape((-1, latent_dim))
        labels = kmeans.predict(data)

        per_task_sample_num = max_episode_len * episode_num // 2

        markov_dist_0.init_transitions(labels=labels[:per_task_sample_num])
        markov_dist_1.init_transitions(labels=labels[per_task_sample_num:])

        test_buffer.clear()
        # Collect episodes from Task_0
        env.set_task(task=task_0)
        collect_stationary_trajectories(env=env,
                                        buffer=test_buffer,
                                        episode_num=episode_num // 2,
                                        episode_len=max_episode_len,
                                        env_change_freq=episode_num)

        # collect episode from Task_1
        env.set_task(task=task_1)
        collect_stationary_trajectories(env=env,
                                        buffer=test_buffer,
                                        episode_num=episode_num // 2,
                                        episode_len=max_episode_len,
                                        env_change_freq=episode_num)

        obs_0, actions_0, rewards_0, next_obs_0 = test_buffer.sample_section(start_idx=0,
                                                                             end_idx=episode_num // 2)
        obs_1, actions_1, rewards_1, next_obs_1 = test_buffer.sample_section(start_idx=episode_num // 2,
                                                                             end_idx=episode_num)

        obs_2 = torch.cat([obs_0[:, :config.train_buffer.max_episode_len // 2, :],
                           obs_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        actions_2 = torch.cat([actions_0[:, :config.train_buffer.max_episode_len // 2, :],
                               actions_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        rewards_2 = torch.cat([rewards_0[:, :config.train_buffer.max_episode_len // 2, :],
                               rewards_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        next_obs_2 = torch.cat([next_obs_0[:, :config.train_buffer.max_episode_len // 2, :],
                                next_obs_1[:, :config.train_buffer.max_episode_len // 2, :]], dim=1)

        # joint Trajectory
        obs_2_d, actions_2_d, rewards_2_d, next_obs_2_d = all_to_device(obs_2, actions_2, rewards_2, next_obs_2, device=config.device)
        latent_sample_2, latent_mean_2, latent_logvar_2, output_0, hidden_state = model.encoder(obs=obs_2_d, actions=actions_2_d, rewards=rewards_2_d)

        sample_joint_trajectory = latent_mean_2[0, ...]
        sample_joint_trajectory = sample_joint_trajectory.detach().cpu().numpy()

        curr_labels = kmeans.predict(sample_joint_trajectory)
        curr_transitions = np.stack([curr_labels[:-1], curr_labels[1:]], axis=1)
        n_c, g_k = run_cusum(curr_transitions, markov_dist_0, markov_dist_1)

        # Single Trajectory
        obs_0_d, actions_0_d, rewards_0_d, next_obs_0_d = all_to_device(obs_0, actions_0, rewards_0, next_obs_0, device=config.device)
        latent_sample_0, latent_mean_0, latent_logvar_0, output_0, hidden_state = model.encoder(obs=obs_0_d, actions=actions_0_d, rewards=rewards_0_d)

        sample_joint_trajectory = latent_mean_0[0, ...]
        sample_joint_trajectory = sample_joint_trajectory.detach().cpu().numpy()

        curr_labels = kmeans.predict(sample_joint_trajectory)
        curr_transitions = np.stack([curr_labels[:-1], curr_labels[1:]], axis=1)
        run_cusum(curr_transitions, markov_dist_0, markov_dist_1)

        a = 1

if __name__ == '__main__':
    main()