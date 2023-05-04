import numpy as np
import torch
from sklearn.cluster import KMeans

def latent_clustering(latent_mean: np.ndarray,
                     cluster_num: int,
                     rg: np.random.RandomState):

    # size of batch_size X seq_len X latent
    # General euclidean clustering of all states from all distributions
    batch_size, seq_len, latent_dim = latent_mean.shape

    # reshape to (-1, latent_dim) --> size will be samples X latent_dim
    data = latent_mean.reshape((-1, latent_dim))
    kmeans = KMeans(n_clusters=cluster_num, random_state=rg).fit(data)

    return kmeans


def latent_clustering_flattened(latent_mean: np.ndarray,
                                cluster_num: int,
                                rg: np.random.RandomState):

    kmeans = KMeans(n_clusters=cluster_num, random_state=rg).fit(latent_mean)

    return kmeans

