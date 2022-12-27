import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import EncoderConfig


class RNNEncoder(nn.Module):

    def __init__(self,
                encoder_config: EncoderConfig,
                state_dim: int = None,
                action_dim: int = None,
                reward_dim: int = None):

        super(RNNEncoder, self).__init__()

        self.encoder_config: EncoderConfig = encoder_config
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.reward_dim: int = reward_dim

        self.recurrent_input_dim = encoder_config.obs_embed_dim + \
                                   encoder_config.action_embed_dim + \
                                   encoder_config.reward_embed_dim

        self.vae_latent_dim = encoder_config.vae_hidden_dim
        self.recurrent_hidden_dim = encoder_config.recurrent_hidden_dim

        self.activation = F.relu
        self.state_encoder = nn.Linear(state_dim, encoder_config.obs_embed_dim)
        self.action_encoder = nn.Linear(action_dim, encoder_config.action_embed_dim)
        self.reward_encoder = nn.Linear(reward_dim, encoder_config.reward_embed_dim)

        self.gru = nn.GRU(input_size=self.recurrent_input_dim,
                          hidden_size=encoder_config.recurrent_hidden_dim,
                          num_layers=1)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # output layer
        self.fc_mu = nn.Linear(encoder_config.recurrent_hidden_dim, self.vae_latent_dim)
        self.fc_logvar = nn.Linear(encoder_config.recurrent_hidden_dim, self.vae_latent_dim)

    def reparameterise(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, actions, states, rewards, hidden_state):

        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """

        # shape should be: sequence_len x batch_size x hidden_size
        # extract features for states, actions, rewards

        ha = self.activation(self.action_encoder(actions))
        hs = self.activation(self.state_encoder(states))
        hr = self.activation(self.reward_encoder(rewards))
        h = torch.cat((ha, hs, hr), dim=2)

        output, _ = self.gru(h, hidden_state)
        gru_h = output.clone()

        # outputs
        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)

        latent_sample = self.reparameterise(latent_mean, latent_logvar)

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]

        return latent_sample, latent_mean, latent_logvar, output