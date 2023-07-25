import numpy as np


class TransformerContinuousToDiscreteRandom:

    def __init__(self,
                 continuous_dim,
                 discrete_num,
                 rnd
                 ):
        self.continuous_dim = continuous_dim
        self.discrete_num = discrete_num
        self.rnd = rnd
        self.centers = None
        self.reset()

    def reset(self):
        self.centers = self.rnd.randn(self.discrete_num, self.continuous_dim)

    def transform(self, sample):
        sample_mat = np.repeat(sample.reshape((1, -1)), self.discrete_num, axis=0)
        dist = np.linalg.norm(sample_mat - self.centers, keepdims=False, axis=1)
        return np.argmin(dist)

