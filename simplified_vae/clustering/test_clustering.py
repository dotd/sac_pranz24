from sklearn.cluster import KMeans
import numpy as np

def main():

    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    labels = kmeans.labels_

    transitions = np.stack([labels[:-1], labels[1:]], axis=1)
    transition_mat = np.zeros((2,2))

    np.add.at(transition_mat, (transitions[:,0], transitions[:,1]), 1)

if __name__ == '__main__':
    main()