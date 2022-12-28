import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Config


class StateTransitionDecoder(nn.Module):

    def __init__(self,
                 config: Config,
                 action_dim: int,
                 obs_dim: int):

        super(StateTransitionDecoder, self).__init__()

        self.config = config
        self.encoder_config = config.encoder
        self.decoder_config = config.state_decoder

        self.state_encoder = nn.Linear(obs_dim, self.encoder_config.obs_embed_dim)
        self.action_encoder = nn.Linear(action_dim, self.encoder_config.action_embed_dim)

        curr_input_dim = self.encoder_config.vae_hidden_dim + \
                         self.encoder_config.obs_embed_dim + \
                         self.encoder_config.action_embed_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(self.decoder_config.layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, self.decoder_config.layers[i]))
            curr_input_dim = self.decoder_config.layers[i]

        self.fc_out = nn.Linear(curr_input_dim, obs_dim)

    def forward(self, latent_state, obs, actions):

        obs_embed = F.relu(self.state_encoder(obs))
        actions_embed = F.relu(self.action_encoder(actions))

        h = torch.cat((latent_state, obs_embed, actions_embed), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(self,
                 config: Config,
                 action_dim: int,
                 obs_dim: int):

        super(RewardDecoder, self).__init__()

        self.config = config
        self.encoder_config = config.encoder
        self.decoder_config = config.reward_decoder

        # get state as input and predict reward prob
        self.state_encoder = nn.Linear(obs_dim, self.encoder_config.obs_embed_dim)
        self.action_encoder = nn.Linear(action_dim, self.encoder_config.action_embed_dim)

        curr_input_dim = self.encoder_config.vae_hidden_dim + \
                         self.encoder_config.action_embed_dim + \
                         self.encoder_config.obs_embed_dim * 2

        self.fc_layers = nn.ModuleList([])
        for i in range(len(self.decoder_config.layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, self.decoder_config.layers[i]))
            curr_input_dim = self.decoder_config.layers[i]

        self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, obs, actions, next_obs):

        obs_embed = F.relu(self.state_encoder(obs))
        actions_embed  = F.relu(self.action_encoder(actions))
        next_obs_embed = F.relu(self.state_encoder(next_obs))

        h = torch.cat((latent_state, obs_embed, actions_embed, next_obs_embed), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)



