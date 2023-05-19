import numpy as np
from collections import deque


class SimpleStats:

    def __init__(self,
                 num_states,
                 num_actions,
                 length
                 ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.length = length
        self.stats = np.zeros(shape=(num_states, num_actions, num_states), dtype=np.int)
        if length is not None:
            self.memory = deque(maxlen=length)

    def add_sample(self, sample):
        self.stats[sample[0], sample[1], sample[2]] = + 1
        sample_to_remove = None
        if self.memory is not None and len(self.memory) == self.length:
            sample_to_remove = self.memory[0]
            self.memory.append(sample)
            self.stats[sample_to_remove[0], sample_to_remove[1], sample_to_remove[2]] = - 1
        return sample_to_remove

    def get_probability(self, state, action):
        return self.stats[state, action] / np.sum(self.stats[state, action])

    def get_mdp(self):
        p = self.stats / np.sum(self.stats, axis=2, keepdims=True)
        return p


class SimpleStatsTransition:

    def __init__(self,
                 num_states,
                 num_actions,
                 length):
        self.simple_stats = [SimpleStats(num_states, num_actions, length // 2),
                             SimpleStats(num_states, num_actions, length // 2)]

    def add_sample(self, sample):
        sampled_removed = self.simple_stats[0].add_sample(sample)
        if sampled_removed is not None:
            self.simple_stats[1].add_sample(sampled_removed)


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
        self.simple_stats_transition = SimpleStatsTransition(num_states, num_actions, length)

    def add_sample(self, sample):
        self.total_samples += 1
        self.simple_stats_transition.add_sample(sample)

