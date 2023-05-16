import numpy as np
from typing import Tuple, List, Union, Optional
from collections import deque
from sklearn.cluster import KMeans


class MarkovDistribution:

    def __init__(self,
                 state_num: int,
                 window_length: int):

        self.state_num: int = state_num
        self.window_length = window_length
        self.dist_window_queue = deque(maxlen=window_length)

        self.transition_mat: np.ndarray = np.zeros((state_num, state_num))
        self.column_sum_vec: np.ndarray = np.zeros((state_num, 1))
        self.update_num = 0

    @property
    def full(self):
        return (len(self.dist_window_queue) == self.window_length)

    def clear(self):

        self.transition_mat: np.ndarray = np.zeros((self.state_num, self.state_num))
        self.column_sum_vec: np.ndarray = np.zeros((self.state_num, 1))

    def pdf(self, sample: Union[Tuple[Tuple, Tuple], List[List]]):
        if self.transition_mat[sample[0], sample[1]] == 0: # No info on transition
            return 0.000001
        else:
            return self.transition_mat[sample[0], sample[1]] / self.column_sum_vec[sample[0], 0]
            # return max(self.transition_mat[sample[0], sample[1]], 0.000001) / min(max(self.column_sum_vec[sample[0], 0], 0.000001), 10000000)

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

    def update_transitions(self, curr_transitions: List):

        for curr_transition in curr_transitions:
            self.update_transition(curr_transition)

    @property
    def probability_mat(self):
        return self.transition_mat / self.column_sum_vec

    def init_transitions(self, labels: np.ndarray):

        curr_transition_mat = np.zeros((self.state_num, self.state_num))

        np.add.at(curr_transition_mat, (labels[0:-1], labels[1:]), 1)

        curr_column_sum_vec = np.sum(curr_transition_mat, axis=1)[:,np.newaxis]

        self.transition_mat = curr_transition_mat
        self.column_sum_vec = curr_column_sum_vec

        curr_transitions = np.stack([labels[0:-1], labels[1:]], axis=1)
        self.dist_window_queue.extend(curr_transitions)

    def reset(self):

        self.transition_mat: np.ndarray = np.zeros((self.state_num, self.state_num))
        self.column_sum_vec: np.ndarray = np.zeros((self.state_num, 1))

        self.dist_window_queue.clear()


class MarkovDistribution3D:

    def __init__(self,
                 state_num: List[int],
                 window_length: int):

        self.state_num: List[int] = state_num
        self.window_length = window_length
        self.dist_window_queue = deque(maxlen=window_length)

        self.transition_mat: np.ndarray = np.zeros((state_num[0], state_num[1], state_num[2]))  # obs, action, next_obs
        self.column_sum_vec: np.ndarray = np.zeros((state_num[0], state_num[1], 1))
        self.update_num = 0

    @property
    def full(self):
        return (len(self.dist_window_queue) == self.window_length)

    def clear(self):

        self.transition_mat: np.ndarray = np.zeros((self.state_num[0], self.state_num[1], self.state_num[2]))
        self.column_sum_vec: np.ndarray = np.zeros((self.state_num[0], self.state_num[1], 1))

    def pdf(self, sample: Union[Tuple[Tuple, Tuple], List[List]]):
        if self.transition_mat[sample[0], sample[1], sample[2]] == 0: # No info on transition
            return 0.000001
        else:
            return self.transition_mat[sample[0], sample[1], sample[2]] / self.column_sum_vec[sample[0], sample[1]]

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

    def update_transitions(self, curr_transitions: List):

        for curr_transition in curr_transitions:
            self.update_transition(curr_transition)

    @property
    def probability_mat(self):
        return self.transition_mat / self.column_sum_vec

    def init_transitions(self,
                         obs_labels: np.ndarray,
                         actions_labels: np.ndarray,
                         next_obs_labels: np.ndarray):

        curr_transition_mat = np.zeros((self.state_num[0], self.state_num[1], self.state_num[2]))

        np.add.at(curr_transition_mat, (obs_labels, actions_labels, next_obs_labels), 1)

        curr_column_sum_vec = np.sum(curr_transition_mat, axis=2, keepdims=True)

        self.transition_mat = curr_transition_mat
        self.column_sum_vec = curr_column_sum_vec

        curr_transitions = np.stack([obs_labels, actions_labels, next_obs_labels], axis=1)
        self.dist_window_queue.extend(curr_transitions)

    def reset(self):

        self.transition_mat: np.ndarray = np.zeros((self.state_num[0], self.state_num[1], self.state_num[2]))
        self.column_sum_vec: np.ndarray = np.zeros((self.state_num[0], self.state_num[1], 1))

        self.dist_window_queue.clear()
