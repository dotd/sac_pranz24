import math
import numpy as np
from typing import Tuple, List, Union

import torch
from sklearn.cluster import KMeans


class MarkovDistribution:

    def __init__(self,
                 state_num: int,
                 clustering: KMeans = None):

        self.state_num: int = state_num
        self.clustering: KMeans = clustering
        self.transition_mat: np.ndarray = np.zeros((state_num, state_num))
        self.column_sum_vec: np.ndarray = np.zeros((state_num))
        self.update_num = 0

    def clear(self):

        self.transition_mat: np.ndarray = np.zeros((self.state_num, self.state_num))
        self.column_sum_vec: np.ndarray = np.zeros((self.state_num))

    def pdf(self, sample: Union[Tuple[Tuple, Tuple], List[List]]):
        return self.transition_mat[sample[0], sample[1]]

    def rvs(self, size: int):
        raise NotImplementedError

    def update_transition(self, oldest_transition, curr_transition):

        self.transition_mat[oldest_transition[0], oldest_transition[1]] -= 1
        self.transition_mat[curr_transition[0], curr_transition[1]] += 1

        self.column_sum_vec[oldest_transition[0]] -= 1
        self.column_sum_vec[curr_transition[0]] += 1

    @property
    def probability_mat(self):
        return self.transition_mat / self.column_sum_vec

    def update_transitions(self, trajectories: Union[np.ndarray, torch.Tensor]):

        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.detach().cpu().numpy()

        curr_transition_mat = np.zeros((self.state_num, self.state_num))

        episode_num, seq_len, hidden_dim = trajectories.shape
        for episode_idx in range(episode_num):

            curr_trajectory = trajectories[episode_idx, :]

            # input should be samples_num X sample_dim
            curr_labels = self.clustering.predict(curr_trajectory)
            curr_transitions = np.stack([curr_labels[:-1], curr_labels[1:]], axis=1)

            np.add.at(curr_transition_mat, (curr_transitions[:, 0], curr_transitions[:, 1]), 1)

        curr_transition_mat /= np.sum(curr_transition_mat, axis=1)[:,np.newaxis]

        self.transition_mat = self.transition_mat + (curr_transition_mat - self.transition_mat) / (self.update_num + 1)
        self.update_num += 1
        self.transition_mat = np.maximum(self.transition_mat, np.ones_like(self.transition_mat) * 0.000001)

