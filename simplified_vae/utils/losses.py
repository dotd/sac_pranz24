import torch


def compute_state_reconstruction_loss(mext_obs_pred: torch.Tensor, next_obs: torch.Tensor):
    """ Compute state reconstruction loss.
    (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

    state_loss = (mext_obs_pred - next_obs).pow(2).mean(dim=-1)
    return state_loss


def compute_reward_reconstruction_loss(rewards_pred: torch.Tensor, rewards: torch.Tensor):
    """ Compute reward reconstruction loss.
    (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

    reward_loss = (rewards_pred - rewards).pow(2).mean(dim=-1)
    return reward_loss


def compute_kl_loss(latent_mean, latent_logvar):

    kl_loss = torch.mean((- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1)))
    return kl_loss


def compute_kl_loss_with_posterior(latent_mean, latent_logvar):

    # -- KL divergence
    gauss_dim = latent_mean.shape[-1]

    # add the gaussian prior
    all_means = torch.cat((torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
    all_logvars = torch.cat((torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))

    # https://arxiv.org/pdf/1811.09975.pdf
    # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
    mu = all_means[1:]
    m = all_means[:-1]
    logE = all_logvars[1:]
    logS = all_logvars[:-1]
    kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
        1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

    return kl_divergences

