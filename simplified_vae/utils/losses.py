import torch


def compute_state_reconstruction_loss(mext_obs_pred: torch.Tensor, next_obs: torch.Tensor):
    """ Compute state reconstruction loss.
    (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

    state_loss = (mext_obs_pred - next_obs).pow(2).mean(dim=-1)

    state_loss = state_loss.sum(dim=-1)
    state_loss = state_loss.sum(dim=-1)
    state_loss = state_loss.mean(dim=0)

    return state_loss


def compute_reward_reconstruction_loss(rewards_pred: torch.Tensor, rewards: torch.Tensor):
    """ Compute reward reconstruction loss.
    (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

    reward_loss = (rewards_pred - rewards).pow(2).mean(dim=-1)

    reward_loss = reward_loss.sum(dim=-1)
    reward_loss = reward_loss.sum(dim=-1)
    reward_loss = reward_loss.mean(dim=0)

    return reward_loss


def compute_kl_loss(latent_mean, latent_logvar):

    kl_loss = torch.mean((- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1)))
    return kl_loss


def compute_kl_loss_with_posterior(latent_mean, latent_logvar):

    # -- KL divergence
    gauss_dim = latent_mean.shape[-1]

    # https://arxiv.org/pdf/1811.09975.pdf
    # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))

    mu = latent_mean[:, 1:, :] # curr_posterior_mean
    m = latent_mean[:, :-1, :] # prev_posterior_mean

    logE = latent_logvar[:, 1:, :] # curr_posterior_logvar
    logS = latent_logvar[:,:-1, :] # prev_posterior_logvar

    kl_loss = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim +
                    torch.sum(torch.exp(logE) / torch.exp(logS), dim=-1) +
                    ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

    kl_loss = kl_loss.sum(dim=-1)
    kl_loss = kl_loss.sum(dim=-1)
    kl_loss = kl_loss.mean(dim=0)

    return kl_loss

