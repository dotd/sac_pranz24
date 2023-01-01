from typing import List

import numpy as np
import torch

from simplified_vae.config.config import Config, BufferConfig


class VAEBuffer(object):

    def __init__(self,
                 config: BufferConfig,
                 obs_dim: int = None,
                 action_dim: int = None):

        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """

        self.config = config

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.device: str = 'cpu'

        self.curr_insert_idx: int = 0  # at which index we're currently inserting new data
        self.is_buffer_full: bool = False

        # buffers for completed episodes (stored on CPU) each buffer is batch_num X seq_len X internal_dim
        self.obs: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, obs_dim)).to(self.device)
        self.actions: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, action_dim)).to(self.device)
        self.rewards: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, 1)).to(self.device)
        self.next_obs: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, obs_dim)).to(self.device)
        self.dones: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, 1)).to(self.device)

    def insert(self, obs: np.ndarray,
                     actions: np.ndarray,
                     rewards: np.ndarray,
                     next_obs: np.ndarray,
                     dones: np.ndarray):

        # TODO support different size of trajectories

        self.obs[self.curr_insert_idx, :, :] = torch.from_numpy(obs).to(self.device)
        self.actions[self.curr_insert_idx, :, :] = torch.from_numpy(actions).to(self.device)
        self.rewards[self.curr_insert_idx, :, :] = torch.from_numpy(rewards).to(self.device)
        self.next_obs[self.curr_insert_idx, :, :] = torch.from_numpy(next_obs).to(self.device)
        self.dones[self.curr_insert_idx, :, :] = torch.from_numpy(dones).to(self.device)

        self.curr_insert_idx = (self.curr_insert_idx + 1) % self.config.max_episode_num

        if self.curr_insert_idx == self.config.max_episode_num - 1:
            self.is_buffer_full = True

    def sample_batch(self, batch_size: int = 5):

        curr_episode_num = self.config.max_episode_num if self.is_buffer_full else self.curr_insert_idx

        batch_size = min(curr_episode_num, batch_size)

        # select the indices for the processes from which we pick
        trajectory_idx = np.random.choice(range(curr_episode_num), batch_size, replace=False)

        # select the rollouts we want Batch X seq_len X internal_dim
        obs = self.obs[trajectory_idx, :, :]
        actions = self.actions[trajectory_idx, :, :]
        rewards = self.rewards[trajectory_idx, :, :]
        next_obs = self.next_obs[trajectory_idx, :, :]

        return obs, actions, rewards, next_obs

    def sample_section(self, start_idx: int, end_idx: int):

        # select the rollouts we want Batch X seq_len X internal_dim
        obs = self.obs[start_idx:end_idx, :, :]
        actions = self.actions[start_idx:end_idx, :, :]
        rewards = self.rewards[start_idx:end_idx, :, :]
        next_obs = self.next_obs[start_idx:end_idx, :, :]

        return obs, actions, rewards, next_obs

    def clear(self):

        self.curr_insert_idx = 0
        self.is_buffer_full = False

        # buffers for completed episodes (stored on CPU) each buffer is batch_num X seq_len X internal_dim
        self.obs: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, self.obs_dim)).to(self.device)
        self.actions: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, self.action_dim)).to(self.device)
        self.rewards: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, 1)).to(self.device)
        self.next_obs: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, self.obs_dim)).to(self.device)
        self.dones: torch.Tensor = torch.zeros((self.config.max_episode_num, self.config.max_episode_len, 1)).to(self.device)