from typing import Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

class Clusterer:

    def __init__(self,
                 cluster_num: int,
                 rg: np.random.RandomState):

        self.cluster_num = cluster_num
        self.rg = rg
        self.clusters: Union[KMeans, MiniBatchKMeans] = None

    # def cluster(self, latent_means):
    #
    #     latent_means = latent_means.detach().cpu().numpy()
    #
    #     # size of batch_size X seq_len X latent
    #     # General euclidean clustering of all states from all distributions
    #     batch_size, seq_len, latent_dim = latent_means.shape
    #
    #     # reshape to (-1, latent_dim) --> size will be samples X latent_dim
    #     data = latent_means.reshape((-1, latent_dim))
    #     self.cluters = KMeans(n_clusters=self.cluster_num, random_state=self.rg).fit(data)

    def cluster(self, latent_means: torch.Tensor):

        latent_means = latent_means.detach().cpu().numpy()

        # size of batch_size X seq_len X latent
        # General euclidean clustering of all states from all distributions
        batch_size, seq_len, latent_dim = latent_means.shape

        # reshape to (-1, latent_dim) --> size will be samples X latent_dim
        data = latent_means.reshape((-1, latent_dim))
        self.clusters = MiniBatchKMeans(n_clusters=self.cluster_num, random_state=self.rg).fit(data)

    def predict(self, latent_means: torch.Tensor):

        if isinstance(latent_means, torch.Tensor):
            latent_means = latent_means.detach().cpu().numpy()
        batch_size, seq_len, latent_dim = latent_means.shape

        # reshape to (-1, latent_dim) --> size will be samples X latent_dim
        data = latent_means.reshape((-1, latent_dim))
        return self.clusters.predict(data)

    def update_clusters(self, latent_means: torch.Tensor):

        if isinstance(latent_means, torch.Tensor):
            latent_means = latent_means.detach().cpu().numpy()

        batch_size, seq_len, latent_dim = latent_means.shape
        data = latent_means.reshape((-1, latent_dim))
        self.clusters = self.clusters.partial_fit(data)