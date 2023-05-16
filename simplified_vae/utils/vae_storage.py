from typing import List

import numpy as np
import torch


class Buffer(object):

    def __init__(self,
                 max_total_steps: int,
                 obs_dim: int = None,
                 action_dim: int = None):

        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """

        self.max_total_steps = max_total_steps

        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.device: str = 'cpu'

        self.curr_steps_count: int = 0
        self.curr_insert_idx: int = 0  # at which index we're currently inserting new data
        self.is_buffer_full: bool = False

        self.obs: List = []
        self.actions: List = []
        self.rewards: List = []
        self.next_obs: List = []
        self.dones: List = []

    def __len__(self):
        return len(self.obs)

    def insert(self, obs: np.ndarray,
                     actions: np.ndarray,
                     rewards: np.ndarray,
                     next_obs: np.ndarray,
                     dones: np.ndarray):

        if self.is_buffer_full:

            self.curr_insert_idx = self.curr_insert_idx % len(self.obs)
            self.curr_steps_count -= len(self.obs[self.curr_insert_idx])
            self.curr_steps_count += len(obs)

            self.obs[self.curr_insert_idx] = obs
            self.actions[self.curr_insert_idx] = actions
            self.rewards[self.curr_insert_idx] = rewards
            self.next_obs[self.curr_insert_idx] = next_obs
            self.dones[self.curr_insert_idx] = dones

            self.curr_insert_idx += 1
        else:

            self.obs.append(obs)
            self.actions.append(actions)
            self.rewards.append(rewards)
            self.next_obs.append(next_obs)
            self.dones.append(dones)

            self.curr_steps_count += len(obs)
            self.curr_insert_idx += 1

        if self.curr_steps_count >= self.max_total_steps:
            self.is_buffer_full = True
        else:
            self.is_buffer_full = False

    def sample_batch(self, batch_size: int = 5):

        batch_size = min(self.curr_insert_idx, batch_size)

        trajectory_idx = np.random.choice(range(self.curr_insert_idx), batch_size, replace=False)

        sort_func = lambda x: len(x[1])
        sorted_idx = sorted([(idx, self.obs[idx]) for idx in trajectory_idx], key=sort_func, reverse=True)
        sorted_idx = [curr[0] for curr in sorted_idx]

        obs = [torch.from_numpy(self.obs[idx]) for idx in sorted_idx]
        actions = [torch.from_numpy(self.actions[idx]) for idx in sorted_idx]
        rewards = [torch.from_numpy(self.rewards[idx]) for idx in sorted_idx]
        next_obs = [torch.from_numpy(self.next_obs[idx]) for idx in sorted_idx]
        lengths = [obs[i].shape[0] for i in range(len(obs))]

        padded_obs = torch.nn.utils.rnn.pad_sequence(obs, batch_first=True, padding_value=0)
        padded_actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True, padding_value=0)
        padded_rewards = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True, padding_value=0)
        padded_next_obs = torch.nn.utils.rnn.pad_sequence(next_obs, batch_first=True, padding_value=0)

        return padded_obs, padded_actions, padded_rewards, padded_next_obs, lengths

    def sample_section(self, start_idx: int, end_idx: int):

        sort_func = lambda x: len(x[1])
        sorted_idx = sorted([(idx, self.obs[idx]) for idx in range(start_idx, end_idx)], key=sort_func, reverse=True)
        sorted_idx = [curr[0] for curr in sorted_idx]

        obs = [self.obs[idx] for idx in sorted_idx]
        actions = [self.actions[idx] for idx in sorted_idx]
        rewards = [self.rewards[idx] for idx in sorted_idx]
        next_obs = [self.next_obs[idx] for idx in sorted_idx]
        lengths = [obs[i].shape[0] for i in range(len(obs))]

        # obs = [torch.from_numpy(self.obs[idx]) for idx in sorted_idx]
        # actions = [torch.from_numpy(self.actions[idx]) for idx in sorted_idx]
        # rewards = [torch.from_numpy(self.rewards[idx]) for idx in sorted_idx]
        # next_obs = [torch.from_numpy(self.next_obs[idx]) for idx in sorted_idx]
        # lengths = [obs[i].shape[0] for i in range(len(obs))]

        return obs, actions, rewards, next_obs, lengths

    def sample_section_padded_seq(self, start_idx: int, end_idx: int):

        sort_func = lambda x: len(x[1])
        sorted_idx = sorted([(idx, self.obs[idx]) for idx in range(start_idx, end_idx)], key=sort_func, reverse=True)
        sorted_idx = [curr[0] for curr in sorted_idx]

        obs = [torch.from_numpy(self.obs[idx]) for idx in sorted_idx]
        actions = [torch.from_numpy(self.actions[idx]) for idx in sorted_idx]
        rewards = [torch.from_numpy(self.rewards[idx]) for idx in sorted_idx]
        next_obs = [torch.from_numpy(self.next_obs[idx]) for idx in sorted_idx]
        lengths = [obs[i].shape[0] for i in range(len(obs))]

        padded_obs = torch.nn.utils.rnn.pad_sequence(obs, batch_first=True, padding_value=0)
        padded_actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True, padding_value=0)
        padded_rewards = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True, padding_value=0)
        padded_next_obs = torch.nn.utils.rnn.pad_sequence(next_obs, batch_first=True, padding_value=0)

        return padded_obs, padded_actions, padded_rewards, padded_next_obs, lengths

    def clear(self):

        self.curr_insert_idx = 0
        self.curr_steps_count = 0
        self.is_buffer_full = False

        # buffers for completed episodes (stored on CPU) each buffer is batch_num X seq_len X internal_dim
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []