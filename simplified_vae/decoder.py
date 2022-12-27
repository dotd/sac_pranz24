import torch
import torch.nn as nn
from torch.nn import functional as F

from config import StateDecoderConfig, EncoderConfig, RewardDecoderConfig


class StateTransitionDecoder(nn.Module):

    def __init__(self,
                 encoder_config: EncoderConfig,
                 decoder_config: StateDecoderConfig,
                 action_dim,
                 state_dim):

        super(StateTransitionDecoder, self).__init__()

        self.state_encoder = nn.Linear(state_dim, encoder_config.obs_embed_dim)
        self.action_encoder = nn.Linear(action_dim, encoder_config.action_embed_dim)

        curr_input_dim = encoder_config.vae_hidden_dim + \
                         encoder_config.obs_embed_dim + \
                         encoder_config.action_embed_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(decoder_config.layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, decoder_config.layers[i]))
            curr_input_dim = decoder_config.layers[i]

        self.fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, actions):

        ha = F.relu(self.action_encoder(actions))
        hs = F.relu(self.state_encoder(state))

        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(self,
                 encoder_config: EncoderConfig,
                 decoder_config: RewardDecoderConfig,
                 action_dim,
                 state_dim):

        super(RewardDecoder, self).__init__()

        # get state as input and predict reward prob
        self.state_encoder = nn.Linear(state_dim, encoder_config.obs_embed_dim)
        self.action_encoder = nn.Linear(action_dim, encoder_config.action_embed_dim)

        curr_input_dim = encoder_config.vae_hidden_dim + \
                         encoder_config.action_embed_dim + \
                         encoder_config.obs_embed_dim * 2

        self.fc_layers = nn.ModuleList([])
        for i in range(len(decoder_config.layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, decoder_config.layers[i]))
            curr_input_dim = decoder_config.layers[i]

        self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, next_state, curr_state, actions):

        hcs = F.relu(self.state_encoder(curr_state))
        ha  = F.relu(self.action_encoder(actions))
        hns = F.relu(self.state_encoder(next_state))

        h = torch.cat((latent_state, hcs, ha, hns), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)



