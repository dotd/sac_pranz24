import math

import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from simplified_vae.clustering.cluter_utils import latent_clustering, latent_clustering_flattened
from simplified_vae.config.config import BaseConfig, ModelConfig
from simplified_vae.config.envs_config import StationaryHopperWindvelEnvConfig, StationaryCheetahWindvelEnvConfig
from simplified_vae.env.environment_factory import env_factory
from simplified_vae.utils.markov_dist import MarkovDistribution, MarkovDistribution3D
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

    # config = BaseConfig(env=StationaryHopperWindvelEnvConfig(),
    #                     model=ModelConfig(checkpoint_path='runs/RNNVAE_Hopper-v3_2023-05-02_15-58-56/model_best.pth.tar'))

    config = BaseConfig(env=StationaryCheetahWindvelEnvConfig(),
                        model=ModelConfig(checkpoint_path='runs/RNNVAE_Hopper-v3_2023-05-02_15-58-56/model_best.pth.tar'))

    rg = set_seed(config.seed)

    # Init Env
    env = env_factory(config=config, logger=None)

    obs_dim: int = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

    max_episode_len = 500
    max_total_steps = 20000
    sample_episode_num = 100 if env.__class__.__name__ == 'StationaryHopperWindVelWrapper' else 10
    clusters_num = [5, 5, 5]
    embeddings_dim = 10

    # state_projection_mat = np.random.normal(size=(obs_dim, embeddings_dim))
    # action_projection_mat = np.random.normal(size=(action_dim, embeddings_dim))

    state_projection_mat = np.eye(obs_dim)
    action_projection_mat = np.eye(action_dim)

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

    obs_0, actions_0, rewards_0, next_obs_0, lengths_0 = task_0_buffer.sample_section(start_idx=0, end_idx=len(task_0_buffer.obs))
    obs_1, actions_1, rewards_1, next_obs_1, lengths_1 = task_1_buffer.sample_section(start_idx=0, end_idx=len(task_1_buffer.obs))

    latent_obs_0 = np.concatenate([curr_obs @ state_projection_mat for curr_obs in obs_0], axis=0)
    latent_obs_1 = np.concatenate([curr_obs @ state_projection_mat for curr_obs in obs_1], axis=0)

    latent_actions_0 = np.concatenate([curr_action @ action_projection_mat for curr_action in actions_0], axis=0)
    latent_actions_1 = np.concatenate([curr_action @ action_projection_mat for curr_action in actions_1], axis=0)

    latent_next_obs_0 = np.concatenate([next_obs @ state_projection_mat for next_obs in next_obs_0], axis=0)
    latent_next_obs_1 = np.concatenate([next_obs @ state_projection_mat for next_obs in next_obs_1], axis=0)

    latent_obs_h = np.concatenate([latent_obs_0, latent_obs_1], axis=0)
    latent_actions_h = np.concatenate([latent_actions_0, latent_actions_1], axis=0)

    obs_cluster = KMeans(n_clusters=clusters_num[0], random_state=rg).fit(latent_obs_h)
    action_cluster = KMeans(n_clusters=clusters_num[1], random_state=rg).fit(latent_actions_h)

    obs_labels_0 = obs_cluster.predict(latent_obs_0)
    obs_labels_1 = obs_cluster.predict(latent_obs_1)

    actions_labels_0 = action_cluster.predict(latent_actions_0)
    actions_labels_1 = action_cluster.predict(latent_actions_1)

    next_obs_labels_0 = obs_cluster.predict(latent_next_obs_0)
    next_obs_labels_1 = obs_cluster.predict(latent_next_obs_1)

    markov_dist_0 = MarkovDistribution3D(state_num=clusters_num,
                                         window_length=max_episode_len)

    markov_dist_1 = MarkovDistribution3D(state_num=clusters_num,
                                         window_length=max_episode_len)

    markov_dist_0.init_transitions(obs_labels=obs_labels_0,
                                   actions_labels=actions_labels_0,
                                   next_obs_labels=next_obs_labels_0)

    markov_dist_1.init_transitions(obs_labels=obs_labels_1,
                                   actions_labels=actions_labels_1,
                                   next_obs_labels=next_obs_labels_1)

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

    obs_0, actions_0, rewards_0, next_obs_0, lengths_0 = task_0_buffer.sample_section(start_idx=0, end_idx=sample_episode_num)
    obs_1, actions_1, rewards_1, next_obs_1, lengths_1 = task_1_buffer.sample_section(start_idx=0, end_idx=sample_episode_num)

    # joint Trajectory
    trajectory_idx = 4

    total_length = [lengths_0[trajectory_idx] + lengths_1[trajectory_idx]]
    obs_2 = np.concatenate([obs_0[trajectory_idx][:lengths_0[trajectory_idx], :],
                            obs_1[trajectory_idx][:lengths_1[trajectory_idx], :]], axis=0)

    actions_2 = np.concatenate([actions_0[trajectory_idx][:lengths_0[trajectory_idx], :],
                                actions_1[trajectory_idx][:lengths_1[trajectory_idx], :]], axis=0)

    rewards_2 = np.concatenate([rewards_0[trajectory_idx][:lengths_0[trajectory_idx], :],
                                rewards_1[trajectory_idx][:lengths_1[trajectory_idx], :]], axis=0)

    next_obs_2 = np.concatenate([next_obs_0[trajectory_idx][:lengths_0[trajectory_idx], :],
                                 next_obs_1[trajectory_idx][:lengths_1[trajectory_idx], :]], axis=0)

    latent_obs = obs_2 @ state_projection_mat
    latent_actions = actions_2 @ action_projection_mat
    latent_next_obs = next_obs_2 @ state_projection_mat

    obs_labels = obs_cluster.predict(latent_obs)
    actions_labels = action_cluster.predict(latent_actions)
    next_obs_labels = obs_cluster.predict(latent_next_obs)

    curr_transitions = np.stack([obs_labels,actions_labels, next_obs_labels], axis=1)
    n_c, g_k = run_cusum(curr_transitions, markov_dist_0, markov_dist_1)


if __name__ == '__main__':
    main()