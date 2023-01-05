from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from simplified_vae.config.config import Config
from torch.autograd import Variable


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 layers: List):

        super().__init__()

        self.activation = F.relu
        self.input_dim = input_dim
        self.ouput_dim = output_dim
        self.layers = layers

        curr_input_dim = input_dim

        self.fc_layers = nn.ModuleList([])
        for i in range(len(self.layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, self.layers[i]))
            curr_input_dim = self.layers[i]

        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, x):

        for i in range(len(self.fc_layers)):
            x = self.activation(self.fc_layers[i](x))

        return self.fc_out(x)


class TWRNET(nn.Module):

    def __init__(self,
                 config: Config,
                 obs_shape: int):

        super().__init__()

        self.config = config
        self.obs_shape = obs_shape

        self.pre, self.post, self.log_std = self._init_distribution_estimation_network()

        # next state proposition
        self.next_state = MLP(input_dim=self.obs_shape + self.config.twr.latents_shape,
                              output_dim=self.obs_shape,
                              layers=self.config.twr.layers)

    def _init_distribution_estimation_network(self):

        # post and pre change parameters
        rand_pre = torch.rand(self.config.twr.latents_shape).to(self.config.device)
        rand_post = rand_pre + 1e-1
        one_latent = torch.ones(size=(1, self.config.twr.latents_shape), requires_grad=True).to(self.config.device)
        one_obs = torch.ones(size=(1, self.config.twr.obs_shape), requires_grad=True).to(self.config.device)

        pre = torch.reshape(2 * rand_pre - 1, shape=(1, self.config.twr.latents_shape))
        post = torch.reshape(2 * rand_post - 1, shape=(1, self.config.twr.latents_shape))

        pre = pre * one_latent
        post = post * one_latent
        log_std = one_obs * self.config.twr.init_std

        return pre, post, log_std

    def distribution_params(self, obs, pre_change):
        """
        distribution of pre_change if pre_change = 1 else: distribution of post change
        """
        obs = obs.to(self.config.device)

        # standard deviation
        std = torch.exp(self.log_std)
        std = torch.clamp(std, 0.01, 1)
        std = std.squeeze(0)  # get to 1D tensor

        # mean
        latent = self.pre if pre_change == 1 else self.post
        latent = latent.repeat(obs.shape[0], 1)

        latent = torch.unsqueeze(latent, dim=1).to(self.config.device) if len(obs.shape) != len(latent.shape) \
            else latent.to(self.config.device)

        input_ = torch.cat((obs, latent), -1)
        mean = self.next_state.forward(input_)

        return mean, std

    def pre_post_distribution(self, previous_obs):

        mean_0, std_0 = self.distribution_params(previous_obs, pre_change=1)
        mean_1, std_1 = self.distribution_params(previous_obs, pre_change=0)

        pre_change_distribution = MultivariateNormal(loc=mean_0, covariance_matrix=torch.diag(std_0))
        post_change_distribution = MultivariateNormal(loc=mean_1, covariance_matrix=torch.diag(std_1))

        return pre_change_distribution, post_change_distribution

    def llr(self, previous_obs, obs):
        """
        compute Log Likelihood Ratio f_\theta_1 / f_\theta_0
        """
        pre_change_distribution, post_change_distribution = self.pre_post_distribution(previous_obs)
        log_likelihood_ratio = post_change_distribution.log_prob(obs) - pre_change_distribution.log_prob(obs)

        return log_likelihood_ratio

    def kl(self, previous_obs):
        """
        compute KL divergence between f_\theta_1 & f_\theta_0(previous_obs)
        """
        previous_obs = previous_obs.to(self.config.device)
        pre_change_distribution, post_change_distribution = self.pre_post_distribution(previous_obs)
        kl = torch.distributions.kl.kl_divergence(pre_change_distribution, post_change_distribution).mean()

        return kl


    def forward(self):
        pass
