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
from simplified_vae.utils.cpd_utils import CPD
from simplified_vae.utils.env_utils import collect_stationary_trajectories
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelWrapper
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, \
    compute_kl_loss, compute_kl_loss_with_posterior
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

        self.cpd = CPD(config=self.config,
                       obs_dim=self.obs_dim,
                       action_dim=self.action_dim,
                       window_length=int(self.config.cpd.cusum_window_length),
                       rg=self.rg)

        # Init Buffer
        self.task_0_buffer = Buffer(max_total_steps=config.cpd.max_total_steps,
                                    obs_dim=self.obs_dim, action_dim=self.action_dim)

        self.task_1_buffer = Buffer(max_total_steps=config.cpd.max_total_steps,
                                    obs_dim=self.obs_dim, action_dim=self.action_dim)

        self.vae_buffer = Buffer(max_total_steps=self.config.vae_train_buffer.max_total_steps,
                                 obs_dim=self.obs_dim, action_dim=self.action_dim)

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

        obs_0_labels, \
        obs_1_labels, \
        actions_0_labels, \
        actions_1_labels,\
        next_obs_labels_0,\
        next_obs_labels_1, = self.cpd.clusterer.init_clusters(obs_0=obs_0,
                                                              obs_1=obs_1,
                                                              actions_0=actions_0,
                                                              actions_1=actions_1,
                                                              next_obs_0=next_obs_0,
                                                              next_obs_1=next_obs_1)

        self.cpd.dists[0].init_transitions(obs_labels=obs_0_labels, actions_labels=actions_0_labels, next_obs_labels=next_obs_labels_0)
        self.cpd.dists[1].init_transitions(obs_labels=obs_1_labels, actions_labels=actions_1_labels, next_obs_labels=next_obs_labels_1)

        # TODO Stabilizes the detection but might affect SAC training
        # self.agents[0].init_replay_buffer(self.task_0_buffer)
        # self.agents[1].init_replay_buffer(self.task_1_buffer)

    def train_model(self):

        print(f'Writing Output to : {self.logger.log_dir}')

        self.curr_agent_idx = 1
        self.env.task_idx = self.curr_agent_idx

        episodes_lengths = [[], []]

        for episode_idx in itertools.count(1):

            curr_episode_total_reward = 0
            curr_episode_steps = 0
            done = False

            obs = self.env.reset()

            all_obs = []
            all_actions = []
            all_next_obs = []
            all_rewards = []
            all_dones= []

            while not done:

                curr_agent = self.agents[self.curr_agent_idx]
                action = self.sample_action(agent=curr_agent, obs=obs)

                if len(curr_agent.replay_memory) > self.config.agent.batch_size:
                    pass
                    # self.update_agent(curr_agent)

                if len(self.vae_buffer) > self.config.training.batch_size:
                    pass
                    # vae_obs, vae_actions, vae_rewards, vae_next_obs, vae_lengths = self.vae_buffer.sample_batch(batch_size=self.config.training.batch_size)
                    #
                    # state_reconstruction_loss, \
                    # reward_reconstruction_loss, \
                    # kl_loss = self.update_vae(obs=vae_obs,
                    #                           actions=vae_actions,
                    #                           rewards=vae_rewards,
                    #                           next_obs=vae_next_obs,
                    #                           lengths=vae_lengths)
                    #
                    # self.log_vae_reward(state_reconstruction_loss,
                    #                     reward_reconstruction_loss,
                    #                     kl_loss,
                    #                     self.total_steps)

                next_obs, reward, done, _ = self.env.step(action)  # Step
                mask = 1 if curr_episode_steps == self.env._max_episode_steps else float(not done)

                curr_agent.replay_memory.push(obs, action, reward, next_obs, mask)  # Append transition to memory

                self.update_cpd(obs=obs,
                                action=action,
                                reward=reward,
                                next_obs=next_obs,
                                lengths=[1],
                                episode_steps=curr_episode_steps,
                                curr_agent_idx=self.curr_agent_idx)

                all_obs.append(obs)
                all_actions.append(action)
                all_next_obs.append(next_obs)
                all_rewards.append(reward)
                all_dones.append(done)

                obs = next_obs
                curr_episode_total_reward += reward
                curr_episode_steps += 1

                self.total_steps += 1
                self.total_agent_steps[self.curr_agent_idx] += 1

                if done:
                    self.log_reward(episode_idx=episode_idx,
                                    episode_steps=curr_episode_steps,
                                    curr_episode_reward=curr_episode_total_reward)
                    episodes_lengths[self.curr_agent_idx].append(curr_episode_steps)

                    self.vae_buffer.insert(np.asarray(all_obs, dtype=np.float32),
                                           np.asarray(all_actions, dtype=np.float32),
                                           np.asarray(all_rewards, dtype=np.float32)[:, np.newaxis],
                                           np.asarray(all_next_obs, dtype=np.float32),
                                           np.asarray(all_dones, dtype=np.float32)[:, np.newaxis])

            if self.total_agent_steps[self.curr_agent_idx] > self.config.agent.num_steps:
                break

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

    def update_cpd(self,
                   obs: Union[np.ndarray, torch.Tensor],
                   action: Union[np.ndarray, torch.Tensor],
                   reward: Union[torch.Tensor, np.ndarray],
                   next_obs: Union[torch.Tensor, np.ndarray],
                   lengths: List,
                   episode_steps: int,
                   curr_agent_idx: int):

        obs_label, \
        action_label, \
        next_obs_label,\
        embedded_obs, \
        embedded_action, \
        embedded_next_obs = self.cpd.clusterer.predict(obs=obs,
                                                       action=action,
                                                       reward=reward,
                                                       next_obs=next_obs,
                                                       lengths=lengths)

        curr_transition = (obs_label, action_label, next_obs_label)
        n_c, g_k = self.cpd.update_transition(embedded_obs=embedded_obs,
                                              embedded_action=embedded_action,
                                              curr_transition=curr_transition,
                                              curr_agent_idx=curr_agent_idx)

        if n_c:  # change has been detected
            self.curr_agent_idx = int(not self.curr_agent_idx)
            self.cpd_detect_counter += 1
            print(f'Change Point Detected at {self.total_steps - (self.config.cpd.cusum_window_length - n_c)}!!!')

        return

    def update_vae(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_obs: torch.Tensor,
                         lengths: List):

        self.model.train()

        obs_d = obs.to(self.config.device)
        actions_d = actions.to(self.config.device)
        rewards_d = rewards.to(self.config.device)
        next_obs_d = next_obs.to(self.config.device)

        if self.config.model.type == 'RNNVAE':
            next_obs_preds, rewards_pred, latent_mean, latent_logvar, _, _ = self.model(obs=obs_d, actions=actions_d, rewards=rewards_d, next_obs=next_obs_d, lengths=lengths)
        elif self.config.model.type == 'VAE':
            next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs=obs_d, actions=actions_d, rewards=rewards_d, next_obs=next_obs_d)
        else:
            raise NotImplementedError

        # TODO next_obs_preds outputs for the padded parts should be zero
        state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs_d)
        reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards_d)

        if self.config.training.use_kl_posterior_loss:
            kl_loss = compute_kl_loss_with_posterior(latent_mean=latent_mean, latent_logvar=latent_logvar)
        else:
            kl_loss = compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar)

        total_loss = self.config.training.state_reconstruction_loss_weight * state_reconstruction_loss + \
                     self.config.training.reward_reconstruction_loss_weight * reward_reconstruction_loss + \
                     self.config.training.kl_loss_weight * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return state_reconstruction_loss.item(), reward_reconstruction_loss.item(), kl_loss.item()

    def log_reward(self, episode_idx, episode_steps, curr_episode_reward):

        self.logger.add_scalar(f'reward_{self.curr_agent_idx}/train', curr_episode_reward, self.total_agent_steps[self.curr_agent_idx])
        print(f'Episode Idx = {episode_idx}, '
              f'Total Steps = {self.total_agent_steps}, {self.total_steps} '
              f'Episode Steps = {episode_steps}, Reward = {curr_episode_reward} '
              f'CPD Window Size = {len(self.cpd.window_queue)}')

    def log_vae_reward(self, state_reconstruction_loss:float,
                             reward_reconstruction_loss: float,
                             kl_loss: float,
                             iter_idx: int):

        self.logger.add_scalar(tag='train/state_reconstruction_loss',
                               scalar_value=state_reconstruction_loss,
                               global_step=iter_idx)
        self.logger.add_scalar(tag='train/reward_reconstruction_loss',
                               scalar_value=reward_reconstruction_loss,
                               global_step=iter_idx)
        self.logger.add_scalar(tag='train/kl_loss',
                               scalar_value=kl_loss,
                               global_step=iter_idx)

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

