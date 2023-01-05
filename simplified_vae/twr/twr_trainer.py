import datetime
import torch

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import Config
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.twr.losses import calc_twr_loss
from simplified_vae.utils.env_utils import sample_non_stationary_trajectory
from simplified_vae.utils.logging_utils import write_config
from simplified_vae.utils.vae_storage import Buffer
from simplified_vae.twr.models import TWRNET


class TWRTrainer:

    def __init__(self,
                 config: Config,
                  env: StationaryCheetahWindVelEnv):

        self.config: Config = config
        self.logger = config.logger
        self.env = env
        self.obs_shape: int = env.observation_space.shape[0]

        self.rg = np.random.RandomState(seed=self.config.seed)
        self.model = TWRNET(config=config, obs_shape=self.obs_shape)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.twr.lr, eps=config.twr.train_eps)

        self.min_loss = np.Inf

        write_config(config=config, logdir=self.logger.log_dir)

    def train_model(self):

        # signal is batch X seq_len X hidden_state
        obs, \
        actions, \
        rewards, \
        next_obs, \
        dones = sample_non_stationary_trajectory(env=self.env,
                                                 max_env_steps=self.config.train_buffer.max_episode_len,
                                                 rg=self.rg)
        signal = obs[0, :, :, np.newaxis]

        kl_list = []  # list of KL divergences
        llr_list = []  # list of log-likelihood ratios
        update_0 = 1  # parameter p_0
        delta = 0  # delay detection for annealing

        for T in range(self.config.twr.t_init, self.config.twr.t_end):
            for _ in range(self.config.twr.n_epochs):

                time_stamps = np.random.choice(T, self.config.twr.batch_size)
                previous_obs = signal[time_stamps]

                obs = signal[time_stamps + 1]
                pre_ts = (time_stamps + delta - T).astype(float).astype(float)
                post_ts = (time_stamps - T).astype(float).astype(float)

                l0, l1 = self.model.update_latents(pre_time_stamps=pre_ts,
                                                   post_time_stamps=post_ts,
                                                   previous_obs=previous_obs,
                                                   obs=obs,
                                                   update_0=update_0)

                kl_list.append(self.model.kl(previous_obs))

            llr = self.model.llr(signal[T - 1], signal[T])
            llr_list.append(llr)

            if self.config.twr.annealing & (update_0 > 0) & (T > self.config.twr.t_min + self.config.twr.t_init):
                if torch.stack(kl_list[-self.config.twr.n_epochs:]).mean() > \
                        torch.stack(kl_list[
                                    self.config.twr.t_min * self.config.twr.n_epochs:-self.config.twr.n_epochs]).mean():  # lower bound = D bar
                    delta += 1
                    update_0 = update_0 - self.config.twr.epsilon

        kl_average = []
        for chunks in [kl_list[x:x + self.config.twr.n_epochs] for x in range(0, len(kl_list), self.config.twr.n_epochs)]:
            kl_average.append(torch.stack(chunks).mean())

        kl_average = torch.stack(kl_average).cpu().detach().numpy()
        llr_flat = (torch.stack(llr_list)).flatten().cpu().detach().numpy()
        return kl_average, llr_flat, kl_list, llr_list

    def train_iter(self, obs: torch.Tensor,
                   actions: torch.Tensor,
                   rewards: torch.Tensor,
                   next_obs: torch.Tensor):

        # self.model.train()
        # self.optimizer.zero_grad()
        #
        # # pre_time_stamps, post_time_stamps, previous_obs, obs, update_0=1):
        # pre_loss = calc_twr_loss(pre_time_stamps, previous_obs, obs, pre_change_loss=1)
        # post_loss = calc_twr_loss(post_time_stamps, previous_obs, obs, pre_change_loss=0)
        #
        # # self.optimizer.zero_grad()
        # if np.random.rand() < update_0:
        #     pre_loss.backward(retain_graph=True)
        #
        # post_loss.backward(retain_graph=True)
        # self.optimizer.step()
        #
        # return pre_loss, post_loss

        # self.optimizer.zero_grad()
        # # total_loss.backward()
        # self.optimizer.step()

        return

    def test_iter(self, obs: torch.Tensor,
                  actions: torch.Tensor,
                  rewards: torch.Tensor,
                  next_obs: torch.Tensor):

        self.model.eval()

        with torch.no_grad():
            pass

