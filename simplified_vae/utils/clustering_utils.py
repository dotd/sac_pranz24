from collections import deque
from typing import Union, List, Callable

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from simplified_vae.config.config import BaseConfig
from simplified_vae.utils.model_utils import init_model


class Clusterer:

    def __init__(self,
                 config: BaseConfig,
                 rg: np.random.RandomState):

        self.config: BaseConfig = config
        self.clusters_num = config.cpd.clusters_num
        self.rg = rg
        self.clusters: Union[KMeans, MiniBatchKMeans] = None

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
            curr_sample = latent_mean_0_flat[np.newaxis, i]
            curr_cluster_idx = labels_0[i]
            self.online_kmeans_queues[curr_cluster_idx].extend(curr_sample)

        sample_num = len(labels_1)
        for i in range(sample_num):
            curr_sample = latent_mean_1_flat[np.newaxis, i]
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

            if len(prev_point.shape) == 2:
                prev_point = np.squeeze(prev_point)

            sample_count = len(self.online_kmeans_queues[curr_label])
            self.clusters.cluster_centers_[curr_label] -= (1.0 / sample_count) * (prev_point - self.clusters.cluster_centers_[curr_label])

        self.online_kmeans_queues[curr_label].append(new_obs[np.newaxis, :])
        sample_count = len(self.online_kmeans_queues[curr_label])
        self.clusters.cluster_centers_[curr_label] += (1.0 / sample_count) * (new_obs - self.clusters.cluster_centers_[curr_label])

    def update_clusters_batch(self, latent_means_vec: List):

        for curr_latent_mean in latent_means_vec:
            self.update_clusters(curr_latent_mean)


class RandomProjectionClusterer:

    def __init__(self,
                 config: BaseConfig,
                 obs_dim: int,
                 action_dim: int,
                 rg: np.random.RandomState):

        self.config: BaseConfig = config

        # self.model, epoch, loss = init_model(config=config,
        #                                      obs_dim=obs_dim,
        #                                      action_dim=action_dim)
        #
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)

        self.clusters_num = config.cpd.clusters_num
        self.rg = rg
        self.obs_cluster: Union[KMeans, MiniBatchKMeans] = None
        self.action_cluster: Union[KMeans, MiniBatchKMeans] = None

        self.state_projection_mat = np.random.normal(size=(obs_dim, self.config.model.encoder.vae_hidden_dim))
        self.action_projection_mat = np.random.normal(size=(action_dim, self.config.model.encoder.vae_hidden_dim))

        self.online_kmeans_obs_queues: List[deque] = [deque(maxlen=self.config.cpd.clusters_queue_size) for _ in range(self.clusters_num[0])]
        self.online_kmeans_action_queues: List[deque] = [deque(maxlen=self.config.cpd.clusters_queue_size) for _ in range(self.clusters_num[1])]

    def init_clusters(self, obs_0: Union[torch.Tensor, np.ndarray],
                            obs_1: Union[torch.Tensor, np.ndarray],
                            actions_0: Union[torch.Tensor, np.ndarray],
                            actions_1: Union[torch.Tensor, np.ndarray],
                            next_obs_0: Union[torch.Tensor, np.ndarray],
                            next_obs_1: Union[torch.Tensor, np.ndarray]):

        latent_obs_0 = np.concatenate([curr_obs @ self.state_projection_mat for curr_obs in obs_0], axis=0)
        latent_obs_1 = np.concatenate([curr_obs @ self.state_projection_mat for curr_obs in obs_1], axis=0)

        latent_actions_0 = np.concatenate([curr_action @ self.action_projection_mat for curr_action in actions_0], axis=0)
        latent_actions_1 = np.concatenate([curr_action @ self.action_projection_mat for curr_action in actions_1], axis=0)

        latent_next_obs_0 = np.concatenate([next_obs @ self.state_projection_mat for next_obs in next_obs_0], axis=0)
        latent_next_obs_1 = np.concatenate([next_obs @ self.state_projection_mat for next_obs in next_obs_1], axis=0)

        latent_obs_h = np.concatenate([latent_obs_0, latent_obs_1], axis=0)
        latent_actions_h = np.concatenate([latent_actions_0, latent_actions_1], axis=0)

        self.obs_cluster = KMeans(n_clusters=self.clusters_num[0], random_state=self.rg).fit(latent_obs_h)
        self.action_cluster = KMeans(n_clusters=self.clusters_num[1], random_state=self.rg).fit(latent_actions_h)

        obs_labels_0 = self.obs_cluster.predict(latent_obs_0)
        obs_labels_1 = self.obs_cluster.predict(latent_obs_1)

        actions_labels_0 = self.action_cluster.predict(latent_actions_0)
        actions_labels_1 = self.action_cluster.predict(latent_actions_1)

        next_obs_labels_0 = self.obs_cluster.predict(latent_next_obs_0)
        next_obs_labels_1 = self.obs_cluster.predict(latent_next_obs_1)

        # sample_num = len(labels_0)
        # for i in range(sample_num):
        #     curr_sample = latent_mean_0_flat[np.newaxis, i]
        #     curr_cluster_idx = labels_0[i]
        #     self.online_kmeans_queues[curr_cluster_idx].extend(curr_sample)
        #
        # sample_num = len(labels_1)
        # for i in range(sample_num):
        #     curr_sample = latent_mean_1_flat[np.newaxis, i]
        #     curr_cluster_idx = labels_1[i]
        #     self.online_kmeans_queues[curr_cluster_idx].extend(curr_sample)

        return obs_labels_0, obs_labels_1, \
               actions_labels_0, actions_labels_1, \
               next_obs_labels_0, next_obs_labels_1

    def cluster(self, latent_means):

        if isinstance(latent_means, torch.Tensor):
            latent_means = latent_means.detach().cpu().numpy()

        self.clusters = KMeans(n_clusters=self.clusters_num, random_state=self.rg).fit(latent_means)

    def predict(self, obs: Union[torch.Tensor, np.ndarray],
                      action: Union[torch.Tensor, np.ndarray],
                      reward: Union[torch.Tensor, np.ndarray],
                      next_obs: Union[torch.Tensor, np.ndarray],
                      lengths: List):

        # TODO self.encode_transition()

        obs_label, latent_obs = self.predict_obs(obs)
        action_label, latent_action = self.predict_action(action)
        next_obs_label, latent_next_obs = self.predict_obs(next_obs)

        return obs_label, \
               action_label, \
               next_obs_label, \
               latent_obs, \
               latent_action, \
               latent_next_obs

    def predict_obs(self, curr_obs):
        latent_obs = (curr_obs @ self.state_projection_mat).reshape(1, -1)
        obs_label = self.obs_cluster.predict(latent_obs)

        return obs_label.item(), latent_obs

    def predict_action(self, action):
        latent_action = (action @ self.action_projection_mat).reshape(1, -1)
        action_label = self.action_cluster.predict(latent_action)

        return action_label.item(), latent_action

    def encode_transition(self, obs: torch.Tensor,
                                action: torch.Tensor,
                                reward: torch.Tensor,
                                hidden_state: torch.Tensor,
                                lengths: List):

        pass

    def update_clusters(self, embedded_obs: Union[torch.Tensor, np.ndarray],
                              embedded_action: Union[torch.Tensor, np.ndarray],
                              obs_label: int,
                              action_label: int):
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

        self.update_cluster(curr_cluster=self.obs_cluster,
                            curr_queues=self.online_kmeans_obs_queues,
                            curr_embedded_sample=embedded_obs,
                            curr_sample_label=obs_label)

        self.update_cluster(curr_cluster=self.action_cluster,
                            curr_queues=self.online_kmeans_action_queues,
                            curr_embedded_sample=embedded_action,
                            curr_sample_label=action_label)

    def update_cluster(self, curr_cluster: KMeans,
                             curr_queues: List[deque],
                             curr_embedded_sample: Union[np.ndarray, torch.Tensor],
                             curr_sample_label: int):

        if isinstance(curr_embedded_sample, torch.Tensor):
            curr_embedded_sample = curr_embedded_sample.squeeze().detach().cpu().numpy()

        if len(curr_queues[curr_sample_label]) == self.config.cpd.clusters_queue_size:
            prev_point = curr_queues[curr_sample_label].popleft()

            if len(prev_point.shape) == 2:
                prev_point = np.squeeze(prev_point)

            sample_count = len(curr_queues[curr_sample_label])
            curr_cluster.cluster_centers_[curr_sample_label] -= (1.0 / sample_count) * (prev_point - curr_cluster.cluster_centers_[curr_sample_label])

        curr_queues[curr_sample_label].append(curr_embedded_sample)
        sample_count = len(curr_queues[curr_sample_label])
        curr_cluster.cluster_centers_[curr_sample_label] += (1.0 / sample_count) * (curr_embedded_sample.squeeze() - curr_cluster.cluster_centers_[curr_sample_label])


