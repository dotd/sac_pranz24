from twr.config import TWRConfig
from twr.models import TWRNET

import torch

class TWRTrainer():

    def __init__(self, config: TWRConfig):

        self.config = config
        self.model = TWRNET(config=config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, eps=config.train_eps)

    def train_model(self):
        a = 1


    def train_iter(self, obs: torch.Tensor,
                   actions: torch.Tensor,
                   rewards: torch.Tensor,
                   next_obs: torch.Tensor):

        self.model.train()


        self.optimizer.zero_grad()
        # total_loss.backward()
        self.optimizer.step()

        return

    def test_iter(self, obs: torch.Tensor,
                  actions: torch.Tensor,
                  rewards: torch.Tensor,
                  next_obs: torch.Tensor):

        self.model.eval()

        with torch.no_grad():
            pass

