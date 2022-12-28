import torch

from simplified_vae.config import Config
from simplified_vae.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, compute_kl_loss
from simplified_vae.vae import VAE
from simplified_vae.vae_storage import VAEBuffer


class VAETrainer:

    def __init__(self, config: Config,
                       buffer: VAEBuffer = None,
                       obs_dim: int = None,
                       action_dim: int = None):

        self.config: Config = config
        self.model = VAE(config=config,
                         obs_dim=obs_dim,
                         action_dim=action_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)

        self.buffer: VAEBuffer = buffer
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim

    def train_model(self):

        self.model.train()
        iter_num = self.config.training.pretrain_episodes // self.config.training.batch_size

        for iter_idx in range(iter_num):

            obs, actions, rewards, next_obs = self.buffer.sample_batch(self.config.training.batch_size)

            self.train_iter(obs, actions, rewards, next_obs)

    def train_iter(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_obs: torch.Tensor):

        next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs, actions, rewards, next_obs)

        state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs)
        reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards)
        kl_loss = compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar)

        total_loss = self.config.training.state_reconstruction_loss_weight * state_reconstruction_loss + \
                     self.config.training.reward_reconstruction_loss_weight * reward_reconstruction_loss + \
                     self.config.training.kl_loss_weight * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

