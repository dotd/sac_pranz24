import itertools
from typing import Union

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import Config
from simplified_vae.env.toggle_windvel_env import ToggleWindVelEnv
from simplified_vae.utils.clustering_utils import Clusterer
from simplified_vae.utils.cpd_utils import CPD
from simplified_vae.utils.env_utils import collect_stationary_trajectories, collect_non_stationary_trajectories, \
    collect_toggle_trajectories
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, \
    compute_kl_loss
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import Buffer
from simplified_vae.utils.logging_utils import write_config
from simplified_vae.models.sac import SAC
from src.utils.replay_memory import ReplayMemory


class POCTrainer:

    def __init__(self, config: Config,
                       env: Union[StationaryCheetahWindVelEnv, ToggleWindVelEnv],
                       data_collection_env: Union[StationaryCheetahWindVelEnv, ToggleWindVelEnv]):

        self.config: Config = config
        self.logger: SummaryWriter = config.logger

        self.env: Union[StationaryCheetahWindVelEnv, ToggleWindVelEnv] = env
        self.data_collection_env: StationaryCheetahWindVelEnv = data_collection_env
        self.obs_dim: int = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

        self.rg = np.random.RandomState(seed=self.config.seed)

        self.model, epoch, loss = init_model(config=config,
                                             obs_dim=self.obs_dim,
                                             action_dim=self.action_dim)

        self.agents = [SAC(config=config,
                           num_inputs=self.obs_dim,
                           action_space=env.action_space) for _ in range(config.agent.agents_num)]

        cpd_num = len(self.config.cpd.window_lengths)
        self.cpds = [CPD(cpd_config=self.config.cpd,
                         window_length=self.config.cpd.window_lengths[i]) for i in range(cpd_num)]

        self.clusterer = Clusterer(config=self.config, rg=self.rg)

        # Init Buffer
        self.buffer = Buffer(max_episode_num=config.train_buffer.max_episode_num,
                             max_episode_len=config.train_buffer.max_episode_len,
                             obs_dim=self.obs_dim,
                             action_dim=self.action_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)
        write_config(config=config, logdir=self.logger.log_dir)

    def init_clusters(self):

        collect_toggle_trajectories(env=self.data_collection_env,
                                    buffer=self.buffer,
                                    episode_num=self.config.train_buffer.max_episode_num,
                                    episode_len=self.config.train_buffer.max_episode_len,
                                    tasks=self.env.tasks,
                                    actor_model=self.agents[0].policy,
                                    device=self.config.device)

        obs_d, actions_d, rewards_d = all_to_device(self.buffer.obs,
                                                    self.buffer.actions,
                                                    self.buffer.rewards,
                                                    device=self.config.device)

        latent_sample, latent_mean, latent_logvar, output_0 = self.model.encoder(obs=obs_d,
                                                                                 actions=actions_d,
                                                                                 rewards=rewards_d)

        # latent_means = self.batched_latent_representation(self.buffer.obs, self.buffer.actions, self.buffer.rewards)

        self.clusterer.cluster(latent_means=latent_mean)

        task_num = len(self.env.tasks)
        per_task_sample_num = self.config.train_buffer.max_episode_len * self.config.train_buffer.max_episode_num // task_num
        all_labels = self.clusterer.predict(latent_means=latent_mean)

        for cpd in self.cpds:
            cpd.dist_0.init_transitions(labels=all_labels[:per_task_sample_num])
            cpd.dist_1.init_transitions(labels=all_labels[per_task_sample_num:])



    def train_model(self):

        total_steps = 0
        updates = 0

        for i_episode in itertools.count(1):

            episode_reward = 0
            episode_steps = 0
            done = False
            obs = self.env.reset()
            prev_label = None
            active_agent_idx = 0

            while not done:

                curr_agent = self.agents[active_agent_idx]

                if self.config.agent.start_steps > total_steps:
                    action = self.env.action_space.sample()  # Sample random action
                else:
                    action = curr_agent.select_action(obs)  # Sample action from policy

                if len(curr_agent.replay_memory) > self.config.agent.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.config.agent.updates_per_step):

                        # Update parameters of all the networks
                        critic_1_loss, \
                        critic_2_loss, \
                        policy_loss, \
                        ent_loss, \
                        alpha = curr_agent.update_parameters(memory=curr_agent.replay_memory,
                                                             batch_size=self.config.agent.batch_size,
                                                             updates=updates)

                        self.logger.add_scalar('agent_loss/critic_1', critic_1_loss, updates)
                        self.logger.add_scalar('agent_loss/critic_2', critic_2_loss, updates)
                        self.logger.add_scalar('agent_loss/policy', policy_loss, updates)
                        self.logger.add_scalar('agent_loss/entropy_loss', ent_loss, updates)
                        self.logger.add_scalar('entropy_temprature/alpha', alpha, updates)
                        updates += 1

                next_obs, reward, done, _ = self.env.step(action)  # Step

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == self.env._max_episode_steps else float(not done)
                curr_agent.replay_memory.push(obs, action, reward, next_obs, mask)  # Append transition to memory

                # update CPD estimation
                curr_label, n_c, g_k = self.update_cpd(obs=obs,
                                                       action=action,
                                                       reward=reward,
                                                       prev_label=prev_label,
                                                       episode_steps=episode_steps)
                # Update policy if CPD is detected
                active_agent_idx = self.update_policy(n_c, active_agent_idx, episode_steps=episode_steps)

                obs = next_obs
                prev_label = curr_label
                episode_steps += 1
                total_steps += 1
                episode_reward += reward

            if total_steps > self.config.agent.num_steps:
                break

            self.logger.add_scalar('reward/train', episode_reward, i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_steps,
                                                                                          episode_steps,
                                                                                          round(episode_reward, 2)))

    def train_iter(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_obs: torch.Tensor):

        pass

    def test_model(self):

        episode_steps = 0
        done = False

        # self.env.set_task(None)
        obs = self.env.reset()
        prev_label = None

        while not done:

            action = self.env.action_space.sample()  # Sample random action

            next_obs, reward, done, _ = self.env.step(action)  # Step

            curr_latent_sample, \
            curr_latent_mean,\
            curr_latent_logvar, \
            curr_output_0 = self.model.encoder(obs=obs,
                                               actions=action,
                                               rewards=np.array([reward]))

            curr_label = self.clusterer.predict(curr_latent_mean)
            print(f'curr label = {curr_label}')
            if episode_steps > 0:
                n_c, g_k = self.cpds[0].update_transition((prev_label.item(), curr_label.item()))
                if n_c:
                    self.add_meta_distribution(self.cpds[0])

                # curr_cpd_estim = [cpd.update_transition((prev_label, curr_label)) for cpd in self.cpds]

            if episode_steps == 80:
                a = 1
            done = done or (episode_steps >= self.config.train_buffer.max_episode_len)
            obs = next_obs
            prev_label = curr_label
            episode_steps += 1

    def update_cpd(self,
                   obs: torch.Tensor,
                   action: torch.Tensor,
                   reward: torch.Tensor,
                   prev_label: np.ndarray,
                   episode_steps: int):

        # perform add_transition and cusum
        curr_latent_sample, \
        curr_latent_mean, \
        curr_latent_logvar, \
        curr_output = self.model.encoder(obs=obs,
                                         actions=action,
                                         rewards=np.array([reward]))

        self.clusterer.update_clusters(new_obs=curr_latent_mean)
        curr_label = self.clusterer.predict(curr_latent_mean)

        print(f'curr label = {curr_label}')
        if episode_steps > 0:
            n_c, g_k = self.cpds[0].update_transition((prev_label.item(), curr_label.item()))
        else:
            n_c, g_k = None, None

        return curr_label, n_c, g_k

    def update_policy(self, n_c, active_agent_idx: int, episode_steps: int):

            if n_c: # change has been detected
                active_agent_idx = int(not active_agent_idx)

            else: # no change, update current transition matrix
                prev_matrix = self.agents[active_agent_idx].transition_mat
                curr_matrix = self.cpds[0].dist_0.transition_mat
                self.agents[active_agent_idx].transition_mat = prev_matrix + (curr_matrix - prev_matrix) / (episode_steps + 1)

            return active_agent_idx

    def test_iter(self, obs: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_obs: torch.Tensor):

        self.model.eval()

        with torch.no_grad():

            obs_d = obs.to(self.config.device)
            actions_d = actions.to(self.config.device)
            rewards_d = rewards.to(self.config.device)
            next_obs_d = next_obs.to(self.config.device)

            next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs_d, actions_d, rewards_d, next_obs_d)

            state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs_d)
            reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards_d)
            kl_loss = compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar)

            return state_reconstruction_loss.item(), reward_reconstruction_loss.item(), kl_loss.item()

