import math
from typing import Tuple

from collections import deque

from simplified_vae.config.config import CPDConfig
from simplified_vae.utils.markov_dist import MarkovDistribution
from simplified_vae.utils.online_median_filter import RunningMedian


class CPD:

    def __init__(self,
                 cpd_config: CPDConfig,
                 window_length: int):

        self.cpd_config = cpd_config
        self.window_length = window_length

        self.dist_0_queue_len = int(window_length * self.cpd_config.alpha_val)
        self.dist_1_queue_len = int(window_length * (1 - self.cpd_config.alpha_val))
        self.window_queue = deque(maxlen=window_length)

        self.dist_0: MarkovDistribution = MarkovDistribution(state_num=cpd_config.clusters_num, window_length=self.dist_0_queue_len)
        self.dist_1: MarkovDistribution = MarkovDistribution(state_num=cpd_config.clusters_num, window_length=self.dist_1_queue_len)

        self.oldest_transition = None

        self.running_median_g_k = RunningMedian(window=self.cpd_config.median_window_size)

    def update_transition(self, curr_transition: Tuple[int, int]):

        self.window_queue.append(curr_transition)

        if len(self.window_queue) == 50:
            a = 1

        if not self.dist_0.full: # dist_0 is not full
            self.dist_0.update_transition(curr_transition=curr_transition)

        else:
            if not self.dist_1.full: # dist_1 is not full
                self.dist_1.update_transition(curr_transition=curr_transition)

            else: # both queues are full
                dist_1_oldest_transition = self.dist_1.update_transition(curr_transition=curr_transition)
                self.dist_0.update_transition(curr_transition=dist_1_oldest_transition)

        n_c, g_k = self.windowed_cusum() if len(self.window_queue) == self.window_length else None, None

        return n_c, g_k

    def windowed_cusum(self):

        n_c, s_k, S_k, g_k, medians_k = 0, [], [], [], []

        for k in range(len(self.window_queue)):

            curr_sample = self.window_queue[k]

            p_0 = max(self.dist_0.pdf(curr_sample), self.cpd_config.dist_epsilon)
            p_1 = max(self.dist_1.pdf(curr_sample), self.cpd_config.dist_epsilon)

            s_k.append(math.log(p_1 / p_0))
            S_k.append(sum(s_k))

            min_S_k = min(S_k)
            g_k.append(S_k[-1] - min_S_k)

            curr_median = self.running_median_g_k.update(S_k[-1] - min_S_k)
            medians_k.append(curr_median)

            # if g_k[-1] > self.cpd_config.cusum_thresh:
            if self.running_median_g_k.median > self.cpd_config.cusum_thresh:
                n_c = S_k.index(min(S_k))
                # break

        return n_c, g_k


