import numpy as np
from sklearn.cluster import KMeans


class Clusterer:

    def __init__(self,
                 cluster_num: int,
                 rg: np.random.RandomState):

        self.cluster_num = cluster_num
        self.rg = rg
        self.kmeans: KMeans = None

    def cluster(self, latent_means):

        latent_means = latent_means.detach().cpu().numpy()

        # size of batch_size X seq_len X latent
        # General euclidean clustering of all states from all distributions
        batch_size, seq_len, latent_dim = latent_means.shape

        # reshape to (-1, latent_dim) --> size will be samples X latent_dim
        data = latent_means.reshape((-1, latent_dim))
        self.kmeans = KMeans(n_clusters=self.cluster_num, random_state=self.rg).fit(data)


    def predict(self, obs):

        return self.kmeans.predict(obs)