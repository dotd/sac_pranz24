from typing import List

import numpy as np
import torch


class VAEStorage(object):

    def __init__(self, max_trajectory_len: int = 100,
                 max_trajectory_num: int = 10000,
                 obs_dim: int = None,
                 action_dim: int = None,
                 device: int = None,
                 ):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """


        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.device: int = device

        self.max_trajectory_num: int = max_trajectory_num  # maximum buffer len (number of trajectories)
        self.curr_insert_idx: int = 0  # at which index we're currently inserting new data
        self.curr_trajectory_num: int = 0  # how much of the buffer has been filled

        # how long a trajectory can be at max (horizon)
        self.max_trajectory_len: int = max_trajectory_len

        # buffers for completed rollouts (stored on CPU)
        # TODO change to Batch First!!!
        self.prev_state: torch.Tensor = torch.zeros((self.max_trajectory_len, self.max_trajectory_num, obs_dim))
        self.next_state: torch.Tensor = torch.zeros((self.max_trajectory_len, self.max_trajectory_num, obs_dim))
        self.actions: torch.Tensor = torch.zeros((self.max_trajectory_len, self.max_trajectory_num, action_dim))
        self.rewards: torch.Tensor = torch.zeros((self.max_trajectory_len, self.max_trajectory_num, 1))
        self.dones: torch.Tensor = torch.zeros((self.max_trajectory_len, self.max_trajectory_num, 1))

        self.trajectory_lens: List = [0] * self.max_trajectory_num

    def insert(self, prev_states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray, dones: np.ndarray):

        # TODO allow for different size of trajectories

        self.prev_state[:, self.curr_insert_idx] = torch.from_numpy(prev_states).to(self.device)
        self.next_state[:, self.curr_insert_idx] = torch.from_numpy(next_states).to(self.device)
        self.actions[:, self.curr_insert_idx] = torch.from_numpy(actions).to(self.device)
        self.rewards[:, self.curr_insert_idx] = torch.from_numpy(rewards).to(self.device)
        self.dones[:, self.curr_insert_idx] = torch.from_numpy(dones).to(self.device)

        self.curr_insert_idx = (self.curr_insert_idx + 1) % self.max_trajectory_num

    def __len__(self):
        return self.curr_trajectory_num

    def get_batch(self, batchsize: int = 5, replace: bool = False):
        # TODO: check if we can get rid of num_enc_len and num_rollouts (call it batchsize instead)

        batchsize = min(self.curr_trajectory_num, batchsize)

        # select the indices for the processes from which we pick
        trajectory_idx = np.random.choice(range(self.curr_trajectory_num), batchsize, replace=replace)
        # trajectory length of the individual rollouts we picked
        trajectory_lens = np.array(self.trajectory_lens)[trajectory_idx]

        # select the rollouts we want
        prev_obs = self.prev_state[:, trajectory_idx, :]
        next_obs = self.next_state[:, trajectory_idx, :]
        actions = self.actions[:, trajectory_idx, :]
        rewards = self.rewards[:, trajectory_idx, :]

        return prev_obs.to(self.device), \
               next_obs.to(self.device), \
               actions.to(self.device), \
               rewards.to(self.device),\
               trajectory_lens