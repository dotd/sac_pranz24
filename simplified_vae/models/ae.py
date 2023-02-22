import torch
import torch.nn as nn

from simplified_vae.config.config import BaseConfig


class AE(nn.Module):

    def __init__(self,
                 config: BaseConfig,
                 state_dim: int = None,
                 action_dim: int = None,
                 reward_dim: int = 1):

        super(AE, self).__init__()

        self.encoder
