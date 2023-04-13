from collections import deque
from typing import Union, List

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from simplified_vae.config.config import BaseConfig


class Clusterer:

    def __init__(self,
                 config: BaseConfig,
                 rg: np.random.RandomState):

        self.config: BaseConfig = config
        self.clusters_num = config.cpd.clusters_num
        self.rg = rg
        self.clusters: Union[KMeans, MiniBatchKMeans] = None
        self.cluster_counts = np.array(self.clusters_num)

        self.online_queues: List[deque] = [deque(maxlen=self.config.cpd.clusters_queue_size) for _ in range(self.clusters_num)]

    def init_clusters(self, latent_mean_h: np.ndarray):

        self.cluster(latent_means=latent_mean_h)

        all_labels = self.predict(latent_means=latent_mean_h)

        batch_size, seq_len, hidden_dim = latent_mean_h.shape
        samples = latent_mean_h.reshape(-1, hidden_dim)

        for cluster_idx in range(self.config.cpd.clusters_num):
            curr_samples = samples[all_labels == cluster_idx, :]
            self.online_queues[cluster_idx].extend(curr_samples)

        return all_labels

    def cluster(self, latent_means):

        if isinstance(latent_means, torch.Tensor):
            latent_means = latent_means.detach().cpu().numpy()

        # size of batch_size X seq_len X latent
        # General euclidean clustering of all states from all distributions
        batch_size, seq_len, latent_dim = latent_means.shape

        # reshape to (-1, latent_dim) --> size will be samples X latent_dim
        data = latent_means.reshape((-1, latent_dim))
        self.clusters = KMeans(n_clusters=self.clusters_num, random_state=self.rg).fit(data)

    def calc_labels(self, samples: Union[np.ndarray, torch.Tensor]):

        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        batch_size, seq_len, latent_dim = samples.shape

        data = samples.reshape((-1, latent_dim))
        all_labels = self.clusters.predict(data)

        return all_labels

    def predict(self, latent_means: Union[torch.Tensor, np.ndarray]):

        if isinstance(latent_means, torch.Tensor):
            latent_means = latent_means.detach().cpu().numpy()
        batch_size, seq_len, latent_dim = latent_means.shape

        # reshape to (-1, latent_dim) --> size will be samples X latent_dim
        data = latent_means.reshape((-1, latent_dim))
        return self.clusters.predict(data)

    def update_clusters(self, new_obs):
        """
        Does an online k-means update on a single data point.
        Args:
            point - a 1 x d array
            k - integer > 1 - number of clusters
            cluster_means - a k x d array of the means of each cluster
            cluster_counts - a 1 x k array of the number of points in each cluster
        Returns:
            An integer in [0, k-1] indicating the assigned cluster.
        Updates cluster_means and cluster_counts in place.
        For initialization, random cluster means are needed.
        """

        if isinstance(new_obs, torch.Tensor):
            new_obs = new_obs.squeeze().detach().cpu().numpy()

        curr_label = self.clusters.predict(new_obs.reshape(1,-1)).item()

        if len(self.online_queues[curr_label]) == self.config.cpd.clusters_queue_size:
            prev_point = self.online_queues[curr_label].popleft()
            sample_count = len(self.online_queues[curr_label])
            self.clusters.cluster_centers_[curr_label] -= (1.0 / sample_count) * (prev_point - self.clusters.cluster_centers_[curr_label])

        self.online_queues[curr_label].append(new_obs)
        sample_count = len(self.online_queues[curr_label])
        self.clusters.cluster_centers_[curr_label] += (1.0 / sample_count) * (new_obs - self.clusters.cluster_centers_[curr_label])

