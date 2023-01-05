import math

import numpy as np
from collections import deque

from simplified_vae.cusum.cusum_utils import MarkovDistribution


class CPD:

    def __init__(self,
                 state_num: int,
                 window_length: int,
                 alpha_val: float):

        self.window_queue = deque(maxlen=window_length)
        self.alpha_val = alpha_val
        self.cusum_threshold = 10

        self.dist_0: MarkovDistribution = MarkovDistribution(state_num=state_num)
        self.dist_1: MarkovDistribution = MarkovDistribution(state_num=state_num)

        self.oldest_transition = None

    def update_transition(self, curr_transition: np.ndarray):

        if len(self.window_queue) == self.window_queue.maxlen:
            self.oldest_transition = self.window_queue.pop()

        self.window_queue.append(curr_transition)

        self.dist_0.update_transition()
        self.dist_1.update_transition()

        return self.windowed_cusum()

    def windowed_cusum(self):

        n_c, s_k, S_k, g_k = 0, [], [], []

        for k in range(len(self.window_queue)):

            curr_sample = self.window_queue[k]

            p_0 = self.dist_0.pdf(curr_sample)
            p_1 = self.dist_1.pdf(curr_sample)

            s_k.append(math.log(p_1 / p_0))
            S_k.append(sum(s_k))

            min_S_k = min(S_k)
            g_k.append(S_k[-1] - min_S_k)

            if g_k[-1] > self.cusum_threshold:
                n_c = S_k.index(min(S_k))
                # break

        return n_c, g_k


