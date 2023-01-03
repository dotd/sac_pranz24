import math
import numpy as np
from typing import Tuple, List, Union


class MarkovDistribution:

    def __init__(self, state_num: int):

        self.state_num: int = state_num
        self.transition_mat: np.ndarray = np.zeros((state_num, state_num))
        self.update_num = 0

    def pdf(self, sample: Union[Tuple[Tuple, Tuple], List[List, List]]):
        return self.transition_mat[sample[0], sample[1]]

    def rvs(self, size: int):
        raise NotImplementedError

    def update_transitions(self, trajectories: np.ndarray):

        curr_transition_mat = np.zeros((self.state_num, self.state_num))

        episode_num, seq_len, hidden_dim = trajectories.shape
        for episode_idx in range(episode_num):

            curr_labels = trajectories[episode_idx, :]
            transitions = np.stack([curr_labels[:-1], curr_labels[1:]], axis=1)

            np.add.at(curr_transition_mat, (transitions[:, 0], transitions[:, 1]), 1)

        curr_transition_mat /= np.sum(curr_transition_mat, axis=1)

        self.transition_mat = self.transition_mat + (curr_transition_mat - self.transition_mat) / (self.update_num + 1)
        self.update_num += 1


def run_cumsum(samples: np.ndarray,
               distribution_0,
               distribution_1,
               threshold: float):

    # samples are batch X seq_len X hidden_dim
    batch_num, seq_len, hidden_dim = samples.shape

    n_c, s_k, S_k, g_k = 0, [], [], []

    for k in range(batch_num):

        curr_sample = samples[k, :]

        p_0 = distribution_0.pdf(curr_sample)
        p_1 = distribution_1.pdf(curr_sample)

        s_k.append(math.log(p_1 / p_0))
        S_k.append(sum(s_k))

        min_S_k = min(S_k)
        g_k.append(S_k[-1] - min_S_k)

        if g_k[-1] > threshold:
            n_c = S_k.index(min(S_k))
            break

    return n_c