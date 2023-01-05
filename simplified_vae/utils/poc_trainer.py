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

        self.cpds = [CPD(state_num=self.config.cpd.clusters_num,
                         window_length=self.config.cpd.window_lens[i],
                         alpha_val=self.config.cpd.alpha_val) for i in range(len(self.config.cpd.window_lens))]

        self.clusterer = Clusterer(cluster_num=self.config.cpd.clusters_num, rg=self.rg)

        # Init Buffer
        self.buffer = Buffer(max_episode_num=config.train_buffer.max_episode_num,
                             max_episode_len=config.train_buffer.max_episode_len,
                             obs_dim=self.obs_dim,
                             action_dim=self.action_dim)

        self.test_buffer = Buffer(max_episode_num=config.test_buffer.max_episode_num,
                                  max_episode_len=config.test_buffer.max_episode_len,
                                  obs_dim=self.obs_dim,
                                  action_dim=self.action_dim)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)
        write_config(config=config, logdir=self.logger.log_dir)

    def init_clusters(self):

        collect_stationary_trajectories(env=self.env,
                                        buffer=self.buffer,
                                        episode_num=self.config.train_buffer.max_episode_num,
                                        episode_len=self.config.train_buffer.max_episode_len,
                                        env_change_freq=self.config.training.env_change_freq)

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

        self.env.set_task(None)
        obs = self.env.reset()
        prev_label = None

        while not done:

            action = self.env.action_space.sample()  # Sample random action

            next_obs, reward, done, _ = self.env.step(action)  # Step

            if episode_steps > 0:

                # TODO embed to latent before clustering
                curr_latent = self.model.encoder(obs=obs,
                                                 actions=action,
                                                 reward=reward,
                                                 next_obs=next_obs)

                curr_label = self.clusterer.predict(curr_latent)

                curr_cpd_estim = [cpd.update_transition(prev_label, curr_label) for cpd in self.cpds]

            obs = next_obs
            prev_label = curr_label
            episode_steps += 1






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

    def batched_latent_representation(self, obs: torch.Tensor,
                                            actions: torch.Tensor,
                                            rewards: torch.Tensor):

        episode_num, episode_len, _ = obs.shape
        batch_size = 64

        self.model.eval()
        with torch.no_grad():

            all_latent_means = torch.zeros([episode_num,
                                            episode_len,
                                            self.config.model.encoder.vae_hidden_dim])

            for batch_idx in range(episode_num // batch_size):

                curr_obs = obs[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...]
                curr_actions = actions[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...]
                curr_rewards = rewards[(batch_idx * batch_size):((batch_idx + 1) * batch_size), ...]

                curr_obs_d, curr_actions_d, curr_rewards_d = all_to_device(curr_obs,
                                                                           curr_actions,
                                                                           curr_rewards,
                                                                           device=self.config.device)

                curr_latent_sample, curr_latent_mean, curr_latent_logvar, curr_output_0 = self.model.encoder(obs=curr_obs_d,
                                                                                                             actions=curr_actions_d,
                                                                                                             rewards=curr_rewards_d)

                all_latent_means[batch_idx, ...] = curr_latent_mean

        return all_latent_means