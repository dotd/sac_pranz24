import numpy as np
from sklearn.cluster import KMeans


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
        self.samples = list()

    def reset(self):
        self.centers = self.rnd.randn(self.discrete_num, self.continuous_dim) * 0.01

    def add_sample(self, sample):
        self.samples.append(sample)

    def do_kmeans(self):
        samples = np.array(self.samples)
        kmeans = KMeans(n_clusters=self.discrete_num, random_state=0, n_init="auto").fit(samples)
        self.centers = kmeans.cluster_centers_

    def transform(self, sample):
        sample_mat = np.repeat(sample.reshape((1, -1)), self.discrete_num, axis=0)
        dist = np.linalg.norm(sample_mat - self.centers, keepdims=False, axis=1)
        return np.argmin(dist)


