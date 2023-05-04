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

        self.online_kmeans_queues: List[deque] = [deque(maxlen=self.config.cpd.clusters_queue_size) for _ in range(self.clusters_num)]

    def init_clusters(self, latent_mean_0: torch.Tensor,
                            latent_mean_1: torch.Tensor,
                            lengths_0: List[int],
                            lengths_1: List[int]):

        episode_num_0 = len(lengths_0)
        episode_num_1 = len(lengths_1)

        latent_mean_0_h = latent_mean_0.detach().cpu().numpy()
        latent_mean_1_h = latent_mean_1.detach().cpu().numpy()

        latent_mean_0_flat = np.concatenate([latent_mean_0_h[i][:lengths_0[i]] for i in range(episode_num_0)], axis=0)
        latent_mean_1_flat = np.concatenate([latent_mean_1_h[i][:lengths_1[i]] for i in range(episode_num_1)], axis=0)

        latent_mean_h = np.concatenate([latent_mean_0_flat, latent_mean_1_flat], axis=0)
        self.cluster(latent_means=latent_mean_h)

        labels_0 = self.predict(latent_mean_0_flat)
        labels_1 = self.predict(latent_mean_1_flat)

        sample_num = len(labels_0)
        for i in range(sample_num):
            curr_sample = latent_mean_0_flat[i]
            curr_cluster_idx = labels_0[i]
            self.online_kmeans_queues[curr_cluster_idx].extend(curr_sample) # TODO bug in the online kmeans calculation

        sample_num = len(labels_1)
        for i in range(sample_num):
            curr_sample = latent_mean_1_flat[i]
            curr_cluster_idx = labels_1[i]
            self.online_kmeans_queues[curr_cluster_idx].extend(curr_sample)

        return labels_0, labels_1

    def cluster(self, latent_means):

        if isinstance(latent_means, torch.Tensor):
            latent_means = latent_means.detach().cpu().numpy()

        self.clusters = KMeans(n_clusters=self.clusters_num, random_state=self.rg).fit(latent_means)

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

        return self.clusters.predict(latent_means)

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

        if len(self.online_kmeans_queues[curr_label]) == self.config.cpd.clusters_queue_size:
            prev_point = self.online_kmeans_queues[curr_label].popleft()
            sample_count = len(self.online_kmeans_queues[curr_label])
            self.clusters.cluster_centers_[curr_label] -= (1.0 / sample_count) * (prev_point - self.clusters.cluster_centers_[curr_label])

        self.online_kmeans_queues[curr_label].append(new_obs)
        sample_count = len(self.online_kmeans_queues[curr_label])
        self.clusters.cluster_centers_[curr_label] += (1.0 / sample_count) * (new_obs - self.clusters.cluster_centers_[curr_label])

