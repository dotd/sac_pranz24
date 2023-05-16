import math

import numpy as np
import torch
import gym
from matplotlib import pyplot as plt

from simplified_vae.clustering.cluter_utils import latent_clustering, latent_clustering_flattened
from simplified_vae.config.config import BaseConfig, ModelConfig
from simplified_vae.config.envs_config import StationaryHopperWindvelEnvConfig
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.utils.markov_dist import MarkovDistribution
from simplified_vae.utils.env_utils import collect_stationary_trajectories, set_seed
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import Buffer


def run_cusum(curr_transitions, markov_dist_0, markov_dist_1):

    n_c, s_k, S_k, g_k = 0, [], [], []
    sample_len = len(curr_transitions)
    done = False

    curr_total_count = np.sum(markov_dist_0.transition_mat)
    next_total_count = np.sum(markov_dist_1.transition_mat)

    for k in range(sample_len):

        curr_sample = curr_transitions[k, :]

        p_0 = markov_dist_0.pdf(curr_sample)
        p_1 = markov_dist_1.pdf(curr_sample)

        curr_prior = markov_dist_0.transition_mat[curr_sample[0], curr_sample[1]] / curr_total_count
        next_prior = markov_dist_1.transition_mat[curr_sample[0], curr_sample[1]] / next_total_count

        p_0 *= curr_prior
        p_1 *= next_prior

        if (curr_prior <= 0.1 and next_prior <= 0.1):
            g_k.append(g_k[-1])# Pad to keep idx correct
            continue

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

    config = BaseConfig(env=StationaryHopperWindvelEnvConfig(),
                        model=ModelConfig(checkpoint_path='runs/RNNVAE_Hopper-v3_2023-05-02_15-58-56/model_best.pth.tar'))

    rg = set_seed(config.seed)

    # Init Env
    env = env_factory(config=config, logger=None)

    obs_dim: int = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

    max_episode_len = 500
    max_total_steps = 20000
    sample_episode_num = 100
    clusters_num = 5

    model, epoch, loss = init_model(config=config,
                                    obs_dim=obs_dim,
                                    action_dim=action_dim)

    # Init Buffer
    task_0_buffer = Buffer(max_total_steps=max_total_steps,
                           obs_dim=obs_dim, action_dim=action_dim)

    task_1_buffer = Buffer(max_total_steps=max_total_steps,
                           obs_dim=obs_dim, action_dim=action_dim)

    tasks = [np.array([0.16308017, 19.30782]), np.array([1.1980728, 5.800347])]

    env.set_task(task=tasks[0])
    task_0 = env.get_task()
    collect_stationary_trajectories(env=env,
                                    buffer=task_0_buffer,
                                    max_env_steps=max_episode_len,
                                    max_total_steps=max_total_steps,
                                    env_change_freq=max_total_steps,
                                    is_print=True)

    # collect episode from Task_1
    env.set_task(task=tasks[1])
    task_1 = env.get_task()
    collect_stationary_trajectories(env=env,
                                    buffer=task_1_buffer,
                                    max_env_steps=max_episode_len,
                                    max_total_steps=max_total_steps,
                                    env_change_freq=max_total_steps,
                                    is_print=True)

    model.eval()
    with torch.no_grad():
        obs_0, actions_0, rewards_0, next_obs_0, lengths_0 = task_0_buffer.sample_section_padded_seq(start_idx=0, end_idx=len(task_0_buffer.obs))
        obs_1, actions_1, rewards_1, next_obs_1, lengths_1 = task_1_buffer.sample_section_padded_seq(start_idx=0, end_idx=len(task_1_buffer.obs))

        obs_0_d, actions_0_d, rewards_0_d, next_obs_0_d = all_to_device(obs_0, actions_0, rewards_0, next_obs_0, device=config.device)
        obs_1_d, actions_1_d, rewards_1_d, next_obs_1_d = all_to_device(obs_1, actions_1, rewards_1, next_obs_1, device=config.device)

        latent_sample_0, latent_mean_0, latent_logvar_0, output_0, hidden_state = model.encoder(obs=obs_0_d, actions=actions_0_d, rewards=rewards_0_d, lengths=lengths_0)
        latent_sample_1, latent_mean_1, latent_logvar_1, output_1, hidden_state = model.encoder(obs=obs_1_d, actions=actions_1_d, rewards=rewards_1_d, lengths=lengths_1)

        latent_mean_0_h = latent_mean_0.detach().cpu().numpy()
        latent_mean_1_h = latent_mean_1.detach().cpu().numpy()

        latent_mean_0_flat = np.concatenate([latent_mean_0_h[i][:lengths_0[i]] for i in range(len(task_0_buffer.obs))], axis=0)
        latent_mean_1_flat = np.concatenate([latent_mean_1_h[i][:lengths_1[i]] for i in range(len(task_1_buffer.obs))], axis=0)

        latent_mean_h = np.concatenate([latent_mean_0_flat, latent_mean_1_flat], axis=0)

        kmeans = latent_clustering_flattened(latent_mean_h, clusters_num, rg)

        markov_dist_0 = MarkovDistribution(state_num=clusters_num,
                                           window_length=max_episode_len,
                                           clustering=kmeans)
        markov_dist_1 = MarkovDistribution(state_num=clusters_num,
                                           window_length=max_episode_len,
                                           clustering=kmeans)

        labels_0 = kmeans.predict(latent_mean_0_flat)
        labels_1 = kmeans.predict(latent_mean_1_flat)

        markov_dist_0.init_transitions(labels=labels_0)
        markov_dist_1.init_transitions(labels=labels_1)

        task_0_buffer.clear()
        task_1_buffer.clear()

        # Collect episodes from Task_0
        env.set_task(task=task_0)
        collect_stationary_trajectories(env=env,
                                        buffer=task_0_buffer,
                                        max_env_steps=max_episode_len,
                                        max_total_steps=max_total_steps,
                                        env_change_freq=max_total_steps,
                                        is_print=True)

        # collect episode from Task_1
        env.set_task(task=task_1)
        collect_stationary_trajectories(env=env,
                                        buffer=task_1_buffer,
                                        max_env_steps=max_episode_len,
                                        max_total_steps=max_total_steps,
                                        env_change_freq=max_total_steps,
                                        is_print=True)

        obs_0, actions_0, rewards_0, next_obs_0, lengths_0 = task_0_buffer.sample_section_padded_seq(start_idx=0, end_idx=sample_episode_num)
        obs_1, actions_1, rewards_1, next_obs_1, lengths_1 = task_1_buffer.sample_section_padded_seq(start_idx=0, end_idx=sample_episode_num)

        # joint Trajectory
        trajectory_idx = 5
        total_length = [lengths_0[trajectory_idx] + lengths_1[trajectory_idx]]
        obs_2 = torch.cat([obs_0[trajectory_idx, :lengths_0[trajectory_idx], :],
                           obs_1[trajectory_idx, :lengths_1[trajectory_idx], :]], dim=0).unsqueeze(dim=0)

        actions_2 = torch.cat([actions_0[trajectory_idx, :lengths_0[trajectory_idx], :],
                               actions_1[trajectory_idx, :lengths_1[trajectory_idx], :]], dim=0).unsqueeze(dim=0)

        rewards_2 = torch.cat([rewards_0[trajectory_idx, :lengths_0[trajectory_idx], :],
                               rewards_1[trajectory_idx, :lengths_1[trajectory_idx], :]], dim=0).unsqueeze(dim=0)

        next_obs_2 = torch.cat([next_obs_0[trajectory_idx, :lengths_0[trajectory_idx], :],
                                next_obs_1[trajectory_idx, :lengths_1[trajectory_idx], :]], dim=0).unsqueeze(dim=0)

        obs_2_d, actions_2_d, rewards_2_d, next_obs_2_d = all_to_device(obs_2, actions_2, rewards_2, next_obs_2, device=config.device)
        latent_sample_2, latent_mean_2, latent_logvar_2, output_0, hidden_state = model.encoder(obs=obs_2_d, actions=actions_2_d, rewards=rewards_2_d, lengths=total_length)

        sample_joint_trajectory = latent_sample_2.detach().cpu().numpy().squeeze()

        curr_labels = kmeans.predict(sample_joint_trajectory)
        curr_transitions = np.stack([curr_labels[:-1], curr_labels[1:]], axis=1)
        n_c, g_k = run_cusum(curr_transitions, markov_dist_0, markov_dist_1)


if __name__ == '__main__':
    main()