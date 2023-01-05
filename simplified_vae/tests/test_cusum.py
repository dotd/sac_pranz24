import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def main():

    episode_len = 150
    dist_0 = multivariate_normal(mean=[1, 2], cov=[[1, 0], [0, 2]])
    # dist_1 = multivariate_normal(mean=[1, 2], cov=[[1, 0], [0, 2]])
    dist_1 = multivariate_normal(mean=[3, 4], cov=[[3, 0], [0, 4]])

    dist_0_samples = dist_0.rvs(size=episode_len // 2)
    dist_1_samples = dist_1.rvs(size=episode_len // 2)

    all_samples = np.concatenate([dist_0_samples, dist_1_samples], axis=0)
    thresh = 10
    n_c, s_k, S_k, g_k = 0, [], [], []

    for k in range(episode_len):

        curr_sample = all_samples[k, :]

        p_0 = dist_0.pdf(curr_sample)
        p_1 = dist_1.pdf(curr_sample)

        s_k.append(math.log(p_1 / p_0))
        S_k.append(sum(s_k))

        min_S_k = min(S_k)
        g_k.append(S_k[-1] - min_S_k)

        if g_k[-1] > thresh:
            n_c = S_k.index(min(S_k))
            # break

    print(f'n_c = {n_c}')
    plt.figure(), plt.plot(g_k), plt.show(block=True)


if __name__ == "__main__":
    main()