import math
import numpy as np
from typing import Tuple, List, Union, Optional
from collections import deque
import torch
from sklearn.cluster import KMeans

from simplified_vae.utils.clustering_utils import Clusterer


class MarkovDistribution:

    def __init__(self,
                 state_num: int,
                 window_length: int,
                 clustering: KMeans = None):

        self.state_num: int = state_num
        self.window_length = window_length
        self.dist_window_queue = deque(maxlen=window_length)

        self.clustering: KMeans = clustering
        self.transition_mat: np.ndarray = np.ones((state_num, state_num)) * np.finfo(float).eps
        self.column_sum_vec: np.ndarray = np.ones((state_num, 1 )) * np.finfo(float).eps
        self.update_num = 0

    @property
    def full(self):
        return (len(self.dist_window_queue) == self.window_length)

    def clear(self):

        self.transition_mat: np.ndarray = np.zeros((self.state_num, self.state_num))
        self.column_sum_vec: np.ndarray = np.zeros((self.state_num, 1))

    def pdf(self, sample: Union[Tuple[Tuple, Tuple], List[List]]):
        return self.transition_mat[sample[0], sample[1]] / self.column_sum_vec[sample[0], 0]

    def rvs(self, size: int):
        raise NotImplementedError

    def update_transition(self, curr_transition: Tuple[int, int]):

        if not self.full:
            oldest_queue_transition = None
            self.dist_window_queue.append(curr_transition)
            self.transition_mat[curr_transition[0], curr_transition[1]] += 1
            self.column_sum_vec[curr_transition[0], 0] += 1

        else:
            oldest_queue_transition = self.dist_window_queue.popleft()
            self.dist_window_queue.append(curr_transition)

            self.transition_mat[oldest_queue_transition[0], oldest_queue_transition[1]] -= 1
            self.column_sum_vec[oldest_queue_transition[0], 0] -= 1

            self.transition_mat[curr_transition[0], curr_transition[1]] += 1
            self.column_sum_vec[curr_transition[0], 0] += 1

        return oldest_queue_transition

    @property
    def probability_mat(self):
        return self.transition_mat / self.column_sum_vec

    def init_transitions(self, labels: np.ndarray):

        curr_transition_mat = np.zeros((self.state_num, self.state_num))
        np.add.at(curr_transition_mat, (labels[0:-1], labels[1:]), 1)

        curr_column_sum_vec = np.sum(curr_transition_mat, axis=1)[:,np.newaxis]

        self.transition_mat = curr_transition_mat
        self.curr_column_sum_vec = curr_column_sum_vec

