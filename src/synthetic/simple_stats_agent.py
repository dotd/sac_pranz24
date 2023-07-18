import numpy as np
from collections import deque

eps = np.finfo(float).eps


class MDPStats:
    """
    This class gets samples and returns the estimated MDP stats
    """

    def __init__(self,
                 num_states,
                 num_actions,
                 length
                 ):
        """
        This is a basic stats class
        :param num_states: num states of the MDP
        :param num_actions: num actions of the MDP
        :param length: the window length to consider
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.length = length
        self.stats = np.zeros(shape=(num_actions, num_states, num_states), dtype=int)
        if self.length is not None:
            self.memory = deque(maxlen=length)
        self.eps = self.num_states * np.finfo(float).eps

    def add_sample(self, action, state, state_next):
        sample = [action, state, state_next]
        self.stats[action, state, state_next] = self.stats[action, state, state_next] + 1
        sample_to_remove = None
        if self.length is not None and len(self.memory) == self.length:
            sample_to_remove = self.memory[0]
            action_, state_, state_next_ = sample_to_remove
            self.memory.append(sample)
            self.stats[action_, state_, state_next_] = self.stats[action_, state_, state_next_] - 1
        else:
            self.memory.append(sample)
        return sample_to_remove

    def get_probability(self, state, action):
        return self.stats[state, action] / np.sum(self.stats[state, action])

    def get_mdp(self, normalization_value=eps):
        p = (self.stats + normalization_value) / np.sum(self.stats + normalization_value, axis=2, keepdims=True)
        return p


class MDPStatsTransition:

    def __init__(self,
                 num_states,
                 num_actions,
                 length):
        self.mdp_stats = [MDPStats(num_states, num_actions, length // 2),
                          MDPStats(num_states, num_actions, length // 2)]

    def add_sample(self, action, state, state_next):
        sampled_removed = self.mdp_stats[1].add_sample(action, state, state_next)
        if sampled_removed is not None:
            self.mdp_stats[0].add_sample(sampled_removed)

    def get_mdps(self):
        mdps = [self.mdp_stats[i].get_mdp() for i in range(2)]
        return mdps

    def get_log_likelihood(self, eps=0.01):
        mdps = self.get_mdps()
        loglikelihood_vec = list()
        for idx in [0, 1]:
            for sample in self.mdp_stats[idx].memory:
                state, action, _, state_next, _, _ = sample
                p1 = mdps[1][state, action, state_next]
                p0 = mdps[0][state, action, state_next]
                loglikelihood = np.log((p1 + eps) / (p0 + eps))
                loglikelihood_vec.append(loglikelihood)
        return loglikelihood_vec


class SimpleStatsAgent:
    """
    In SimpleStatsAgent out assumptions are as follows:
    1) We do not know from which MDP we start
    2)
    """

    def __init__(self,
                 num_states,
                 num_actions,
                 length):
        self.total_samples = 0
        self.simple_stats_transition = MDPStatsTransition(num_states, num_actions, length)

    def add_sample(self, sample):
        self.total_samples += 1
        self.simple_stats_transition.add_sample(sample)
