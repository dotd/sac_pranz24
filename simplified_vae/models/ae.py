import torch
import torch.nn as nn

from simplified_vae.config.config import Config


class AE(nn.Module):

    def __init__(self,
                 config: Config,
                 state_dim: int = None,
                 action_dim: int = None,
                 reward_dim: int = 1):

        super(AE, self).__init__()

        self.encoder
