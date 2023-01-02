import numpy as np
import sklearn.cluster
import torch
from sklearn.cluster import KMeans

from simplified_vae.config.config import Config


def latent_clustering(latent_mean: np.ndarray,
                     latent_logver: np.ndarray,
                     cluster_num: int,
                     rg: np.random.RandomState):

    # size of batch_size X seq_len X latent

    # General euclidean clustering of all states from all distributions
    batch_size, seq_len, latent_dim = latent_mean.shape

    # reshape to (-1, latent_dim) --> size will be samples X latent_dim
    data = latent_mean.reshape((-1, latent_dim))
    kmeans = KMeans(n_clusters=cluster_num, random_state=rg, n_init="auto").fit(data)

    return kmeans


def create_transition_matrix(config: Config,
                             kmeans: KMeans,
                             clusters_num: int):

    transition_mat = np.zeros((clusters_num, clusters_num))

    data_labels = kmeans.labels_
    data_labels = data_labels.reshape(-1, config.train_buffer.max_episode_len, config.model.encoder.vae_hidden_dim)

    episode_num, seq_len, hidden_dim = data_labels.shape
    for episode_idx in range(episode_num):

        curr_labels = data_labels[episode_idx, :]
        transitions = np.stack([curr_labels[:-1], curr_labels[1:]], axis=1)

        np.add.at(transition_mat, (transitions[:, 0], transitions[:, 1]), 1)

    return transition_mat

