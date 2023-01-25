import math
from typing import Tuple, List

from collections import deque

import numpy as np

from simplified_vae.config.config import CPDConfig, Config
from simplified_vae.utils.markov_dist import MarkovDistribution
from simplified_vae.utils.online_median_filter import RunningMedian


class CPD:

    def __init__(self,
                 config: Config,
                 window_length: int):

        self.config: Config = config
        self.cpd_config: CPDConfig = config.cpd
        self.window_length = window_length

        self.window_queue = deque(maxlen=window_length)

        self.dist_queue_len = self.cpd_config.max_episode_len * self.cpd_config.max_episode_num // 2 - 1

        self.dists: List[MarkovDistribution] = [MarkovDistribution(state_num=self.cpd_config.clusters_num, window_length=self.dist_queue_len),
                                                MarkovDistribution(state_num=self.cpd_config.clusters_num, window_length=self.dist_queue_len)]

        self.oldest_transition = None

    def update_transition(self, curr_transition: Tuple[int, int], curr_agent_idx: int):

        self.window_queue.append(curr_transition)
        self.dists[curr_agent_idx].update_transition(curr_transition=curr_transition)

        if len(self.window_queue) == self.window_length:
           n_c, g_k = self.windowed_cusum(curr_agent_idx)
        else:
            n_c, g_k = None, None

        if n_c:
            # print("Change Point Detected!!!")
            self.window_queue.clear()

        return n_c, g_k

    def windowed_cusum(self, curr_agent_idx: int):

        running_median = RunningMedian(window=self.cpd_config.median_window_size)

        n_c, s_k, S_k, g_k, medians_k = None, [], [], [], []
        next_agent_idx = int(not(curr_agent_idx))
        done = False

        for k in range(len(self.window_queue)):

            curr_sample = self.window_queue[k]

            curr_p = max(self.dists[curr_agent_idx].pdf(curr_sample), self.cpd_config.dist_epsilon)
            next_p = max(self.dists[next_agent_idx].pdf(curr_sample), self.cpd_config.dist_epsilon)

            s_k.append(math.log(next_p / curr_p))
            S_k.append(sum(s_k))

            min_S_k = min(S_k)
            g_k.append(S_k[-1] - min_S_k)

            curr_median = running_median.update(S_k[-1] - min_S_k)
            medians_k.append(curr_median)

            # if g_k[-1] > self.cpd_config.cusum_thresh:
            if running_median.median > self.cpd_config.cusum_thresh and not done:
                n_c = k #S_k.index(min(S_k))
                done = True
                # break

        return n_c, g_k


