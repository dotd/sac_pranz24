import math
from typing import Tuple, List

from collections import deque

import numpy as np
import scipy.signal as signal
from numpy.random import RandomState

from simplified_vae.config.config import CPDConfig, BaseConfig
from simplified_vae.utils.clustering_utils import Clusterer
from simplified_vae.utils.markov_dist import MarkovDistribution
from simplified_vae.utils.online_median_filter import RunningMedian


class CPD:

    def __init__(self,
                 config: BaseConfig,
                 window_length: int,
                 rg: RandomState):

        self.config: BaseConfig = config
        self.cpd_config: CPDConfig = config.cpd
        self.window_length = window_length

        self.window_queue = deque(maxlen=window_length)
        self.latent_means_vec = []

        self.dist_queue_len = self.cpd_config.max_total_steps // 2 - 1

        self.dists: List[MarkovDistribution] = [MarkovDistribution(state_num=self.cpd_config.clusters_num, window_length=self.dist_queue_len),
                                                MarkovDistribution(state_num=self.cpd_config.clusters_num, window_length=self.dist_queue_len)]

        self.clusterer = Clusterer(config=self.config, rg=rg)

        self.oldest_transition = None
        self.cusum_monitoring_sig: bool = True

    def update_transition(self, curr_transition: Tuple[int, int], curr_latent_mean, curr_agent_idx: int):

        self.window_queue.append(curr_transition)
        self.latent_means_vec.append(curr_latent_mean)

        if len(self.window_queue) == self.window_length:

            if self.cusum_monitoring_sig:
                print('Start Cusum Monitoring')
                self.cusum_monitoring_sig = False

            n_c, g_k, medians_k = self.windowed_cusum(curr_agent_idx)
        else:
            n_c, g_k = None, None

        if n_c:
            next_agent_idx = int(not curr_agent_idx)
            self.dists[curr_agent_idx].update_transitions(curr_transitions=list(self.window_queue)[:n_c])
            self.dists[next_agent_idx].update_transitions(curr_transitions=list(self.window_queue)[n_c:])

            self.clusterer.update_clusters_batch(latent_means_vec=self.latent_means_vec)

            self.cusum_monitoring_sig = True
            self.window_queue.clear()
            self.latent_means_vec.clear()

        return n_c, g_k

    def windowed_cusum(self, curr_agent_idx: int):

        running_median = RunningMedian(window=self.cpd_config.median_window_size)

        n_c, s_k, S_k, g_k, medians_k = None, [], [], [], []
        next_agent_idx = int(not(curr_agent_idx))
        done = False

        curr_total_count = np.sum(self.dists[curr_agent_idx].transition_mat)
        next_total_count = np.sum(self.dists[next_agent_idx].transition_mat)

        for k in range(len(self.window_queue)):

            curr_sample = self.window_queue[k]

            curr_p = max(self.dists[curr_agent_idx].pdf(curr_sample), self.cpd_config.dist_epsilon)
            next_p = max(self.dists[next_agent_idx].pdf(curr_sample), self.cpd_config.dist_epsilon)

            curr_prior = max(self.dists[curr_agent_idx].transition_mat[curr_sample] / curr_total_count, self.config.cpd.prior_cusum_eps)
            next_prior = max(self.dists[next_agent_idx].transition_mat[curr_sample] / next_total_count, self.config.cpd.prior_cusum_eps)

            curr_p *= curr_prior
            next_p *= next_prior

            if curr_prior <= 0.1 and next_prior <= 0.1:
                if g_k:
                    g_k.append(g_k[-1])
                    medians_k.append(medians_k[-1])

                continue

            s_k.append(math.log(next_p / curr_p))
            S_k.append(sum(s_k))

            min_S_k = min(S_k)
            g_k.append(S_k[-1] - min_S_k)

            curr_median = running_median.update(S_k[-1] - min_S_k)
            medians_k.append(curr_median)

            if running_median.median > self.cpd_config.cusum_thresh:
                done = True

        if done:
            max_val = max(g_k)
            max_idx = g_k.index(max_val)

            zero_idxs = [idx for idx, val in enumerate(g_k) if abs(val) <= 1.0]
            zeros_diff = [abs(max_idx - curr_zero_idx) for curr_zero_idx in zero_idxs]
            closest_zero_idx = zeros_diff.index(min(zeros_diff))
            n_c = zero_idxs[closest_zero_idx]

            return n_c, g_k, medians_k
        else:
            return None, g_k, medians_k



