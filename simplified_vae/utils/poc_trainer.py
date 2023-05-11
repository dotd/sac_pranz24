import itertools
from collections import deque
from typing import Union, List

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from simplified_vae.config.config import BaseConfig
from simplified_vae.env.fixed_toggle_abs_env import FixedToggleABSEnv
from simplified_vae.env.fixed_toggle_cheetah_windvel_wrapper import FixedToggleCheetahWindVelWrapper
from simplified_vae.env.stationary_abs_env import StationaryABSEnv
from simplified_vae.env.toggle_abs_env import ToggleABSEnv
from simplified_vae.env.toggle_cheetah_windvel_wrapper import ToggleCheetahWindVelWrapper
from simplified_vae.utils.clustering_utils import Clusterer
from simplified_vae.utils.cpd_utils import CPD
from simplified_vae.utils.env_utils import collect_stationary_trajectories, collect_non_stationary_trajectories, \
    collect_toggle_trajectories
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelWrapper
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, \
    compute_kl_loss
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import Buffer
from simplified_vae.utils.logging_utils import write_config
from simplified_vae.models.sac import SAC


class POCTrainer:

    def __init__(self, config: BaseConfig,
                 env: Union[StationaryCheetahWindVelWrapper, ToggleCheetahWindVelWrapper],
                 data_collection_env: Union[StationaryCheetahWindVelWrapper, ToggleCheetahWindVelWrapper],
                 logger: SummaryWriter):

        self.config: BaseConfig = config
        self.logger: SummaryWriter = logger

        self.env: Union[ToggleCheetahWindVelWrapper, FixedToggleCheetahWindVelWrapper, ToggleABSEnv, FixedToggleABSEnv] = env
        self.data_collection_env: Union[StationaryCheetahWindVelWrapper, StationaryABSEnv] = data_collection_env

        self.obs_dim: int = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

        self.rg = np.random.RandomState(seed=self.config.seed)

        self.agents = [SAC(config=config,
                           num_inputs=self.obs_dim,
                           action_space=env.action_space) for _ in range(config.agent.agents_num)]

        self.model, epoch, loss = init_model(config=config,
                                             obs_dim=self.obs_dim,
                                             action_dim=self.action_dim)

        self.cpd = CPD(config=self.config, window_length=int(self.config.cpd.cusum_window_length))

        self.clusterer = Clusterer(config=self.config, rg=self.rg)

        # Init Buffer
        self.task_0_buffer = Buffer(max_total_steps=config.cpd.max_total_steps,
                                    obs_dim=self.obs_dim, action_dim=self.action_dim)

        self.task_1_buffer = Buffer(max_total_steps=config.cpd.max_total_steps,
                                    obs_dim=self.obs_dim, action_dim=self.action_dim)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)

        self.total_agent_steps: List = [0,0]
        self.total_agent_updates: List = [0, 0]

        self.avg_reward_windows = [deque(maxlen=self.config.training.sum_reward_window_size),
                                   deque(maxlen=self.config.training.sum_reward_window_size)]

        self.total_steps = 0
        self.cpd_detect_counter: int = 0
        self.curr_agent_idx = 0

        write_config(config=config, logdir=self.logger.log_dir)

    def init_clusters(self):

        # Collect episodes from Task_0
        self.data_collection_env.set_task(task=self.env.tasks[0])
        collect_stationary_trajectories(env=self.data_collection_env,
                                        buffer=self.task_0_buffer,
                                        max_env_steps=self.config.cpd.max_env_steps,
                                        max_total_steps=self.config.cpd.max_total_steps,
                                        env_change_freq=self.config.cpd.max_total_steps,
                                        agent=self.agents[0],
                                        is_print=True)

        # collect episode from Task_1
        self.data_collection_env.set_task(task=self.env.tasks[1])
        collect_stationary_trajectories(env=self.data_collection_env,
                                        buffer=self.task_1_buffer,
                                        max_env_steps=self.config.cpd.max_env_steps,
                                        max_total_steps=self.config.cpd.max_total_steps,
                                        env_change_freq=self.config.cpd.max_total_steps,
                                        agent=self.agents[1],
                                        is_print=True)

        obs_0, actions_0, rewards_0, next_obs_0, lengths_0 = self.task_0_buffer.sample_section(start_idx=0, end_idx=len(self.task_0_buffer.obs))
        obs_1, actions_1, rewards_1, next_obs_1, lengths_1 = self.task_1_buffer.sample_section(start_idx=0, end_idx=len(self.task_1_buffer.obs))

        obs_0_d, actions_0_d, rewards_0_d, next_obs_0 = all_to_device(obs_0, actions_0, rewards_0, next_obs_0, device=self.config.device)
        obs_1_d, actions_1_d, rewards_1_d, next_obs_1 = all_to_device(obs_1, actions_1, rewards_1, next_obs_1, device=self.config.device)

        self.model.eval()
        with torch.no_grad():

            if self.config.model.type == 'RNNVAE':
                latent_sample_0, latent_mean_0, latent_logvar_0, output_0, hidden_0 = self.model.encoder(obs=obs_0_d,
                                                                                                         actions=actions_0_d,
                                                                                                         rewards=rewards_0_d,
                                                                                                         lengths=lengths_0)

                latent_sample_1, latent_mean_1, latent_logvar_1, output_1, hidden_1 = self.model.encoder(obs=obs_1_d,
                                                                                                         actions=actions_1_d,
                                                                                                         rewards=rewards_1_d,
                                                                                                         lengths=lengths_1)
            elif self.config.model.type == 'VAE':
                latent_sample_0, latent_mean_0, latent_logvar_0 = self.model.encoder(obs=obs_0_d,
                                                                                     actions=actions_0_d,
                                                                                     rewards=rewards_0_d)

                latent_sample_1, latent_mean_1, latent_logvar_1 = self.model.encoder(obs=obs_1_d,
                                                                                     actions=actions_1_d,
                                                                                     rewards=rewards_1_d)
            else:
                raise NotImplementedError

        labels_0, labels_1 = self.clusterer.init_clusters(latent_mean_0=latent_mean_0,
                                                          latent_mean_1=latent_mean_1,
                                                          lengths_0=lengths_0,
                                                          lengths_1=lengths_1)

        self.cpd.dists[0].init_transitions(labels=labels_0)
        self.cpd.dists[1].init_transitions(labels=labels_1)

        self.agents[0].init_replay_buffer(self.task_0_buffer)
        self.agents[1].init_replay_buffer(self.task_1_buffer)

    def train_model(self):

        print(f'Writing Output to : {self.logger.log_dir}')

        self.curr_agent_idx = 1
        self.env.task_idx = self.curr_agent_idx

        episodes_lengths = [[], []]

        for episode_idx in itertools.count(1):

            curr_episode_total_reward = 0
            curr_episode_steps = 0
            done = False
            prev_label = None
            hidden_state = None

            obs = self.env.reset()

            while not done:

                curr_agent = self.agents[self.curr_agent_idx]
                action = self.sample_action(agent=curr_agent, obs=obs)

                if len(curr_agent.replay_memory) > self.config.agent.batch_size:
                    self.update_agent(curr_agent)

                next_obs, reward, done, _ = self.env.step(action)  # Step
                mask = 1 if curr_episode_steps == self.env._max_episode_steps else float(not done)
                curr_agent.replay_memory.push(obs, action, reward, next_obs, mask)  # Append transition to memory

                hidden_state, curr_label, n_c, g_k = self.update_cpd(obs=obs,
                                                                     action=action,
                                                                     reward=reward,
                                                                     hidden_state=hidden_state,
                                                                     lengths=[1],
                                                                     prev_label=prev_label,
                                                                     episode_steps=curr_episode_steps,
                                                                     curr_agent_idx=self.curr_agent_idx)

                if n_c:  # change has been detected
                    self.curr_agent_idx = int(not self.curr_agent_idx)
                    self.cpd_detect_counter += 1
                    print(f'Change Point Detected at {self.total_steps}!!!')
                    # wandb.log({'detected_cpd_step_v1':episode_steps}, step=cpd_detect_counter)
                    # wandb.log({'detected_cpd_step_v2': episode_steps}, step=episode_steps)

                obs = next_obs
                prev_label = curr_label
                curr_episode_total_reward += reward
                curr_episode_steps += 1

                self.total_steps += 1
                self.total_agent_steps[self.curr_agent_idx] += 1

                if done:
                    self.log_reward(episode_idx=episode_idx,
                                    episode_steps=curr_episode_steps,
                                    curr_episode_reward=curr_episode_total_reward)
                    episodes_lengths[self.curr_agent_idx].append(curr_episode_steps)

            if self.total_agent_steps[self.curr_agent_idx] > self.config.agent.num_steps:
                break

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
            # print(f'curr label = {curr_label}')
            if episode_steps > 0:
                n_c, g_k = self.cpd.update_transition((prev_label.item(), curr_label.item()))
                if n_c:
                    self.add_meta_distribution(self.cpd)

                # curr_cpd_estim = [cpd.update_transition((prev_label, curr_label)) for cpd in self.cpds]

            if episode_steps == 80:
                a = 1
            done = done or (episode_steps >= self.config.train_buffer.max_episode_len)
            obs = next_obs
            prev_label = curr_label
            episode_steps += 1

    def update_agent(self, curr_agent):

        # Number of updates per step in environment
        for i in range(self.config.agent.updates_per_step):
            # Update parameters of all the networks
            critic_1_loss, \
            critic_2_loss, \
            policy_loss, \
            ent_loss, \
            alpha = curr_agent.update_parameters(memory=curr_agent.replay_memory,
                                                 batch_size=self.config.agent.batch_size,
                                                 updates=self.total_agent_updates[self.curr_agent_idx])

            self.logger.add_scalar(f'agent_loss_{self.curr_agent_idx}/critic_1', critic_1_loss, self.total_agent_updates[self.curr_agent_idx])
            self.logger.add_scalar(f'agent_loss_{self.curr_agent_idx}/critic_2', critic_2_loss, self.total_agent_updates[self.curr_agent_idx])
            self.logger.add_scalar(f'agent_loss_{self.curr_agent_idx}/policy', policy_loss, self.total_agent_updates[self.curr_agent_idx])
            self.logger.add_scalar(f'agent_loss_{self.curr_agent_idx}/entropy_loss', ent_loss, self.total_agent_updates[self.curr_agent_idx])
            self.logger.add_scalar(f'entropy_temprature_{self.curr_agent_idx}/alpha', alpha, self.total_agent_updates[self.curr_agent_idx])

            # wandb.log({'agent_loss/critic_1': critic_1_loss, 'agent_loss/critic_2': critic_2_loss,
            #            'agent_loss/policy': policy_loss, 'agent_loss/entropy_loss': ent_loss,
            #            'entropy_temprature/alpha': alpha}, step=updates)

            self.total_agent_updates[self.curr_agent_idx] += 1

    def sample_action(self, agent: SAC, obs: np.ndarray):

        if self.config.agent.start_steps > self.total_agent_steps[self.curr_agent_idx]:
            action = self.env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(obs)  # Sample action from policy

        return action

    def encode_transition(self, obs: torch.Tensor,
                                action: torch.Tensor,
                                reward: torch.Tensor,
                                hidden_state: torch.Tensor,
                                lengths: List):

        self.model.eval()
        with torch.no_grad():
            # perform add_transition and cusum

            if self.config.model.type == 'RNNVAE':
                curr_latent_sample, \
                curr_latent_mean, \
                curr_latent_logvar, \
                curr_output, hidden_state = self.model.encoder(obs=obs,
                                                               actions=action,
                                                               rewards=np.array([reward]),
                                                               hidden_state=hidden_state,
                                                               lengths=lengths)

            elif self.config.model.type == 'VAE':
                curr_latent_sample, \
                curr_latent_mean, \
                curr_latent_logvar = self.model.encoder(obs=obs,
                                                        actions=action,
                                                        rewards=np.array([reward]))

            else:
                raise NotImplementedError

        return curr_latent_mean

    def update_cpd(self,
                   obs: torch.Tensor,
                   action: torch.Tensor,
                   reward: torch.Tensor,
                   hidden_state: torch.Tensor,
                   lengths: List,
                   prev_label: np.ndarray,
                   episode_steps: int,
                   curr_agent_idx: int):

        curr_latent_mean = self.encode_transition(obs=obs,
                                                  action=action,
                                                  reward=reward,
                                                  hidden_state=hidden_state,
                                                  lengths=lengths)

        self.clusterer.update_clusters(new_obs=curr_latent_mean)
        curr_label = self.clusterer.predict(curr_latent_mean.reshape(1,-1))

        if episode_steps > 0:

            curr_sample = [prev_label.item(), curr_label.item()]
            next_agent_idx = int(not curr_agent_idx)
            curr_p = self.cpd.dists[curr_agent_idx].pdf(curr_sample)
            next_p = self.cpd.dists[next_agent_idx].pdf(curr_sample)

            if next_p > curr_p:
                a = 1

            n_c, g_k = self.cpd.update_transition(curr_transition=(prev_label.item(), curr_label.item()),
                                                  curr_agent_idx=curr_agent_idx)
        else:
            n_c, g_k = None, None

        return hidden_state, curr_label, n_c, g_k

    def log_reward(self, episode_idx, episode_steps, curr_episode_reward):

        self.logger.add_scalar(f'reward_{self.curr_agent_idx}/train', curr_episode_reward, self.total_agent_steps[self.curr_agent_idx])
        print(f'Episode Idx = {episode_idx}, '
              f'Total Steps = {self.total_agent_steps}, {self.total_steps} '
              f'Episode Steps = {episode_steps}, Reward = {curr_episode_reward} '
              f'CPD Window Size = {len(self.cpd.window_queue)}')

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

