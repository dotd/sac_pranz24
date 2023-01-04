import torch
import torch.nn as nn

from twr.config import TWRConfig


class MLP(nn.Module):

    pass

class TWRNET(nn.Module):

    def __init__(self, config: TWRConfig):

        super().__init__()

        self.config = config
        self.obs_shape = self.config.obs_shape

        self.pre, self.post, self.log_std = self._init_distribution_estimation_network()

        # next state proposition
        self.next_state = MLP()

    def _init_distribution_estimation_network(self):

        # post and pre change parameters
        rand_pre = torch.rand(self.config.latents_shape)
        rand_post = rand_pre + 1e-1
        one_latent = torch.ones(size=(1, self.config.latents_shape), requires_grad=True)
        one_obs = torch.ones(size=(1, self.config.obs_shape), requires_grad=True)

        pre = torch.reshape(2 * rand_pre - 1, shape=(1, self.config.latents_shape))
        post = torch.reshape(2 * rand_post - 1, shape=(1, self.config.latents_shape))

        pre = pre * one_latent
        post = post * one_latent
        log_std = one_obs * self.config.init_std

        return pre, post, log_std
