import math
from typing import Tuple, List

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

        self.window_queue = deque(maxlen=window_length)

        self.dists: List[MarkovDistribution] = [MarkovDistribution(state_num=cpd_config.clusters_num, window_length=self.cpd_config.transition_dist_window_size),
                                                MarkovDistribution(state_num=cpd_config.clusters_num, window_length=self.cpd_config.transition_dist_window_size)]

        # self.dist_0: MarkovDistribution = MarkovDistribution(state_num=cpd_config.clusters_num, window_length=self.cpd_config.transition_dist_window_size)
        # self.dist_1: MarkovDistribution = MarkovDistribution(state_num=cpd_config.clusters_num, window_length=self.cpd_config.transition_dist_window_size)

        self.oldest_transition = None

        self.running_median_g_k = RunningMedian(window=self.cpd_config.median_window_size)

    def update_transition(self, curr_transition: Tuple[int, int], curr_agent_idx: int):

        self.window_queue.append(curr_transition)

        # if not self.dist_0.full: # dist_0 is not full
        #     self.dist_0.update_transition(curr_transition=curr_transition)
        #
        # else:
        #     if not self.dist_1.full: # dist_1 is not full
        #         self.dist_1.update_transition(curr_transition=curr_transition)
        #
        #     else: # both queues are full
        #         dist_1_oldest_transition = self.dist_1.update_transition(curr_transition=curr_transition)
        #         self.dist_0.update_transition(curr_transition=dist_1_oldest_transition)

        next_agent_idx = int(not(curr_agent_idx))
        if not self.dists[curr_agent_idx].full: # curr dist is not full
            self.dists[curr_agent_idx].update_transition(curr_transition=curr_transition)

        else:
            if not self.dists[next_agent_idx].full: # next dist is not full
                self.dists[next_agent_idx].update_transition(curr_transition=curr_transition)

            else: # both queues are full
                oldest_transition = self.dists[next_agent_idx].update_transition(curr_transition=curr_transition)
                self.dists[curr_agent_idx].update_transition(curr_transition=oldest_transition)

        n_c, g_k = self.windowed_cusum(curr_agent_idx) if len(self.window_queue) == self.window_length else (None, None)

        if n_c:
            print("Change Point Detected!!!")
            # Update Queues
            [self.window_queue.popleft() for _ in range(n_c)]
            self.dists[curr_agent_idx].dist_window_queue.clear()

            # Update transition matrices
            # TODO update transition matrices
            self.dists[curr_agent_idx].transition_mat
            self.dists[curr_agent_idx].column_sum_vec

        return n_c, g_k

    def windowed_cusum(self, curr_agent_idx: int):

        n_c, s_k, S_k, g_k, medians_k = None, [], [], [], []
        other_agent_idx = int(not(curr_agent_idx))

        for k in range(len(self.window_queue)):

            curr_sample = self.window_queue[k]

            # p_0 = max(self.dist_0.pdf(curr_sample), self.cpd_config.dist_epsilon)
            # p_1 = max(self.dist_1.pdf(curr_sample), self.cpd_config.dist_epsilon)

            p_0 = max(self.dists[curr_agent_idx].pdf(curr_sample), self.cpd_config.dist_epsilon)
            p_1 = max(self.dists[other_agent_idx].pdf(curr_sample), self.cpd_config.dist_epsilon)

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


