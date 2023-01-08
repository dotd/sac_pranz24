import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from simplified_vae.config.config import EncoderConfig, Config


class RNNEncoder(nn.Module):

    def __init__(self,
                config: Config,
                state_dim: int = None,
                action_dim: int = None,
                reward_dim: int = 1):

        super(RNNEncoder, self).__init__()

        self.config: Config = config
        self.encoder_config: EncoderConfig = config.encoder
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.reward_dim: int = reward_dim

        self.recurrent_input_dim = self.encoder_config.obs_embed_dim + \
                                   self.encoder_config.action_embed_dim + \
                                   self.encoder_config.reward_embed_dim

        self.vae_latent_dim = self.encoder_config.vae_hidden_dim
        self.recurrent_hidden_dim = self.encoder_config.recurrent_hidden_dim

        self.activation = F.relu
        self.state_encoder = nn.Linear(state_dim, self.encoder_config.obs_embed_dim)
        self.action_encoder = nn.Linear(action_dim, self.encoder_config.action_embed_dim)
        self.reward_encoder = nn.Linear(reward_dim, self.encoder_config.reward_embed_dim)

        self.gru = nn.GRU(input_size=self.recurrent_input_dim,
                          hidden_size=self.encoder_config.recurrent_hidden_dim,
                          num_layers=1, batch_first=True)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # output layer
        self.fc_mu = nn.Linear(self.encoder_config.recurrent_hidden_dim, self.vae_latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_config.recurrent_hidden_dim, self.vae_latent_dim)

    def \
            reparameterise(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.config.device)
        z = mu + noise * std

        return z

    def forward(self, obs: torch.Tensor,
                      actions: torch.Tensor,
                      rewards: torch.Tensor,
                      hidden_state: torch.Tensor = None):

        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """
        if len(obs.shape) == 1:
            obs = obs[np.newaxis,np.newaxis,:]
        if len(actions.shape) == 1:
            actions = actions[np.newaxis, np.newaxis, :]
        if len(rewards.shape) == 1:
            rewards = rewards[np.newaxis, np.newaxis,:]

        if isinstance(obs, np.ndarray):
            obs = torch.Tensor(obs).to(self.config.device)
        if isinstance(actions, np.ndarray):
            actions = torch.Tensor(actions).to(self.config.device)
        if isinstance(rewards, np.ndarray):
            rewards = torch.Tensor(rewards).to(self.config.device)



        # shape should be: batch_size X sequence_len X hidden_size
        # extract features for states, actions, rewards
        obs_embed = self.activation(self.state_encoder(obs))
        actions_embed = self.activation(self.action_encoder(actions))
        reward_embed = self.activation(self.reward_encoder(rewards))
        h = torch.cat((actions_embed, obs_embed, reward_embed), dim=2)

        output, _ = self.gru(h, hidden_state)
        gru_h = output.clone()

        # outputs
        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)

        latent_sample = self.reparameterise(latent_mean, latent_logvar)

        return latent_sample, latent_mean, latent_logvar, output