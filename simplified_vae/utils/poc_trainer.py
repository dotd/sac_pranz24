from typing import Union

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import Config
from simplified_vae.models.vae import VAE, RNNVAE
from simplified_vae.utils.clustering_utils import Clusterer
from simplified_vae.utils.cpd_utils import CPD
from simplified_vae.utils.env_utils import collect_stationary_trajectories, collect_non_stationary_trajectories
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, \
    compute_kl_loss, compute_kl_loss_with_posterior
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import Buffer
from simplified_vae.utils.logging_utils import save_checkpoint, write_config


class POCTrainer:

    def __init__(self, config: Config,
                       env: StationaryCheetahWindVelEnv):

        self.config: Config = config
        self.logger: SummaryWriter = config.logger

        self.env: StationaryCheetahWindVelEnv = env
        self.obs_dim: int = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

        self.rg = np.random.RandomState(seed=self.config.seed)

        self.model, epoch, loss = init_model(config=config,
                                             obs_dim=self.obs_dim,
                                             action_dim=self.action_dim)

        cpd_num = len(self.config.cpd.window_lengths)
        self.cpds = [CPD(cpd_config=self.config.cpd,
                         window_length=self.config.cpd.window_lengths[i]) for i in range(cpd_num)]

        self.clusterer = Clusterer(cluster_num=self.config.cpd.clusters_num, rg=self.rg)
        self.meta_distributions: np.ndarray = np.zeros((self.config.cpd.meta_dist_num,
                                                        self.config.cpd.clusters_num,
                                                        self.config.cpd.clusters_num))

        # Init Buffer
        self.buffer = Buffer(max_episode_num=config.train_buffer.max_episode_num,
                             max_episode_len=config.train_buffer.max_episode_len,
                             obs_dim=self.obs_dim,
                             action_dim=self.action_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)
        write_config(config=config, logdir=self.logger.log_dir)

    def init_clusters(self):

        collect_stationary_trajectories(env=self.env,
                                        buffer=self.buffer,
                                        episode_num=self.config.train_buffer.max_episode_num,
                                        episode_len=self.config.train_buffer.max_episode_len,
                                        env_change_freq=self.config.train_buffer.max_episode_num)

        obs_d, actions_d, rewards_d = all_to_device(self.buffer.obs,
                                                    self.buffer.actions,
                                                    self.buffer.rewards,
                                                    device=self.config.device)

        latent_sample, latent_mean, latent_logvar, output_0 = self.model.encoder(obs=obs_d,
                                                                                 actions=actions_d,
                                                                                 rewards=rewards_d)

        # latent_means = self.batched_latent_representation(self.buffer.obs, self.buffer.actions, self.buffer.rewards)

        self.clusterer.cluster(latent_means=latent_mean)

    def train_model(self):

        pass

    def train_iter(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_obs: torch.Tensor):

        pass

    def test_model(self):

        episode_steps = 0
        done = False

        # self.env.set_task(None)
        obs = self.env.reset()
        prev_label = None

        while not done:

            action = self.env.action_space.sample()  # Sample random action

            next_obs, reward, done, _ = self.env.step(action)  # Step

            curr_latent_sample, \
            curr_latent_mean,\
            curr_latent_logvar, \
            curr_output_0 = self.model.encoder(obs=obs,
                                               actions=action,
                                               rewards=np.array([reward]))

            curr_label = self.clusterer.predict(curr_latent_mean)
            print(f'curr label = {curr_label}')
            if episode_steps > 0:
                n_c, g_k = self.cpds[0].update_transition((prev_label.item(), curr_label.item()))
                if n_c:
                    self.add_meta_distribution(self.cpds[0])

                # curr_cpd_estim = [cpd.update_transition((prev_label, curr_label)) for cpd in self.cpds]

            if episode_steps == 80:
                a = 1
            done = done or (episode_steps >= self.config.train_buffer.max_episode_len)
            obs = next_obs
            prev_label = curr_label
            episode_steps += 1

    def add_meta_distribution(self, cpd: CPD):

        dist_0 = cpd.dist_0
        dist_1 = cpd.dist_1

        if self.meta_distributions is None:
            self.meta_distributions[0, :, :] = dist_0
            self.meta_distributions[0, :, :] = dist_1

        # TODO find closest distribution to the new one

    def test_iter(self, obs: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_obs: torch.Tensor):

        self.model.eval()

        with torch.no_grad():

            obs_d = obs.to(self.config.device)
            actions_d = actions.to(self.config.device)
            rewards_d = rewards.to(self.config.device)
            next_obs_d = next_obs.to(self.config.device)

            next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs_d, actions_d, rewards_d, next_obs_d)

            state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs_d)
            reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards_d)
            kl_loss = compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar)

            return state_reconstruction_loss.item(), reward_reconstruction_loss.item(), kl_loss.item()

