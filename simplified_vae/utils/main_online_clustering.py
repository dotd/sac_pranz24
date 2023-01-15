import math

import numpy as np
import torch
import gym
from matplotlib import pyplot as plt

from simplified_vae.clustering.cluter_utils import latent_clustering
from simplified_vae.config.config import Config
from simplified_vae.utils.markov_dist import MarkovDistribution
from simplified_vae.utils.env_utils import make_stationary_env, collect_stationary_trajectories, set_seed
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import Buffer


def run_cusum(curr_transitions, markov_dist_0, markov_dist_1):

    n_c, s_k, S_k, g_k = 0, [], [], []
    sample_len = len(curr_transitions)
    for k in range(sample_len):

        curr_sample = curr_transitions[k, :]

        p_0 = markov_dist_0.pdf(curr_sample)
        p_1 = markov_dist_1.pdf(curr_sample)

        s_k.append(math.log(p_1 / p_0))
        S_k.append(sum(s_k))

        min_S_k = min(S_k)
        g_k.append(S_k[-1] - min_S_k)

        if g_k[-1] > 10:
            n_c = S_k.index(min(S_k))
            # break

    print(f'n_c = {n_c}')
    plt.figure(), plt.plot(g_k), plt.show(block=True)

def init_stage(env,
               test_buffer,
               episode_num,
               max_episode_len,
               model,
               clusters_num,
               rg,
               device):

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
        obs_1, actions_1, rewards_1, next_obs_1 = test_buffer.sample_section(start_idx=episode_num //2,
                                                                             end_idx=episode_num)

        obs_0_d, actions_0_d, rewards_0_d, next_obs_0_d = all_to_device(obs_0, actions_0, rewards_0, next_obs_0, device=device)
        obs_1_d, actions_1_d, rewards_1_d, next_obs_1_d = all_to_device(obs_1, actions_1, rewards_1, next_obs_1, device=device)

        latent_sample_0, latent_mean_0, latent_logvar_0, output_0, hidden_state = model.encoder(obs=obs_0_d, actions=actions_0_d, rewards=rewards_0_d)
        latent_sample_1, latent_mean_1, latent_logvar_1, output_1, hidden_state = model.encoder(obs=obs_1_d, actions=actions_1_d, rewards=rewards_1_d)

        # latent_mean = latent_mean_0 # TODO decide on which data should we do the clustering
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

        return markov_dist_0, markov_dist_1, kmeans

def main():

    ## Init config
    config = Config()
    rg = set_seed(config.seed)

    # Init Env
    env = make_stationary_env(config=config)
    obs_dim: int = env.observation_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]
    episode_num = 2000
    max_episode_len = 100
    clusters_num = 10

    model, epoch, loss = init_model(config=config,
                                    obs_dim=obs_dim,
                                    action_dim=action_dim)

    # Init Buffer
    test_buffer = Buffer(max_episode_num=episode_num,
                         max_episode_len=max_episode_len,
                         obs_dim=obs_dim, action_dim=action_dim)

    init_stage(env=env,
               test_buffer=test_buffer,
               episode_num=episode_num,
               max_episode_len=max_episode_len,
               model=model,
               clusters_num=clusters_num,
               rg=rg,
               device=config.device)

    done = False
    total_steps = 0
    hidden_state = None
    prev_label = None

    # TODO generate a single joint trajectory and pass it through the VAE + CUSUM, compare output to an online version of the same trajectory

    while not done:

        action = env.action_space.sample()  # Sample random action

        next_obs, reward, done, _ = env.step(action)  # Step

        # update CPD estimation
        hidden_state, curr_label, n_c, g_k = self.update_cpd(obs=obs,
                                                             action=action,
                                                             reward=reward,
                                                             hidden_state=hidden_state,
                                                             prev_label=prev_label,
                                                             episode_steps=episode_steps)
        # Update policy if CPD is detected
        active_agent_idx = self.update_policy(n_c, active_agent_idx, episode_steps=episode_steps)

        obs = next_obs
        prev_label = curr_label
        total_steps += 1





if __name__ == '__main__':
    main()