import numpy as np
import torch

from simplified_vae.config.config import Config


def calc_twr_loss(config: Config,
                  time_stamps: np.ndarray,
                  previous_obs,
                  obs,
                  pre_change_loss,
                  alpha=.005,
                  rho=.05):

    """
    loss of pre_change if pre_change_loss = 1 else: loss of post change
    """
    pre_change_distribution, post_change_distribution = pre_post_distribution(previous_obs)
    obs = obs.to(config.device)

    if config.twr.loss_type == 'KL':  # KL divergence
        loss = post_change_distribution.log_prob(obs) - pre_change_distribution.log_prob(obs)
    elif config.twr.loss_type == 'sqrt':
        loss = torch.sqrt(post_change_distribution.prob(obs) / pre_change_distribution.prob(obs))
    else:  # self.config.loss_type == 'exp':
        q = post_change_distribution.prob(obs) / pre_change_distribution.prob(obs)
        loss = (q + 1) * torch.log(q)

    # evaluate asymptotic probabilities parameters: Shirayev case
    kl = torch.distributions.kl.kl_divergence(pre_change_distribution, post_change_distribution).mean()
    m = - np.abs(np.log(alpha)) / (kl + np.abs(np.log(rho)))
    s = np.sqrt(3) * np.abs(np.log(alpha)) / (np.pi * (kl + np.abs(np.log(rho))))

    # evaluate asymptotic probabilities
    t = (torch.from_numpy(time_stamps).to(config.device) - m) / s
    w = (torch.tensor(1) - torch.sigmoid(t)) if pre_change_loss == 1 else -torch.sigmoid(t)
    loss = torch.dot(w, loss.squeeze())

    return loss



