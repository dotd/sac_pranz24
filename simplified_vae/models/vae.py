import warnings

import numpy as np
import torch
import torch.nn as nn

from simplified_vae.models.decoder import StateTransitionDecoder, RewardDecoder
from simplified_vae.models.encoder import RNNEncoder, Encoder

from simplified_vae.config.config import Config


class RNNVAE(nn.Module):

    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, config: Config,
                       obs_dim: int = None,
                       action_dim: int = None):

        # initialise the encoder
        super().__init__()
        self.config = config
        self.encoder = RNNEncoder(config=config,
                                  state_dim=obs_dim,
                                  action_dim=action_dim).to(config.device).to(self.config.device)

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder = StateTransitionDecoder(config=config, action_dim=action_dim, obs_dim=obs_dim).to(self.config.device)
        self.reward_decoder = RewardDecoder(config=config, action_dim=action_dim, obs_dim=obs_dim).to(self.config.device)

    def forward(self,
                obs: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                next_obs: torch.Tensor,
                hidden_state: torch.Tensor = None):

        latent_sample, latent_mean, latent_logvar, output, hidden_state = self.encoder(obs=obs,
                                                                                       actions=actions,
                                                                                       rewards=rewards,
                                                                                       hidden_state=hidden_state)

        next_obs_preds = self.state_decoder(latent_sample, obs, actions)
        rewards_pred = self.reward_decoder(latent_sample, obs, actions, next_obs)

        return next_obs_preds, rewards_pred, latent_mean, latent_logvar, output, hidden_state


class VAE(nn.Module):

    def __init__(self, config: Config,
                       obs_dim: int = None,
                       action_dim: int = None):

        # initialise the encoder
        super().__init__()
        self.config = config
        self.encoder = Encoder(config=config,
                               state_dim=obs_dim,
                               action_dim=action_dim).to(config.device).to(self.config.device)

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder = StateTransitionDecoder(config=config, action_dim=action_dim, obs_dim=obs_dim).to(self.config.device)
        self.reward_decoder = RewardDecoder(config=config, action_dim=action_dim, obs_dim=obs_dim).to(self.config.device)

    def forward(self,
                obs: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                next_obs: torch.Tensor):

        latent_sample, latent_mean, latent_logvar = self.encoder(obs=obs,
                                                                 actions=actions,
                                                                 rewards=rewards)

        next_obs_preds = self.state_decoder(latent_sample, obs, actions)
        rewards_pred = self.reward_decoder(latent_sample, obs, actions, next_obs)

        return next_obs_preds, rewards_pred, latent_mean, latent_logvar