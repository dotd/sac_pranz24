import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import Config
from simplified_vae.utils.env_utils import collect_stationary_train_trajectories, collect_stationary_test_trajectories
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, compute_kl_loss
from simplified_vae.models.vae import VAE
from simplified_vae.utils.vae_storage import VAEBuffer
from simplified_vae.utils.logging_utils import save_checkpoint


class VAETrainer:

    def __init__(self, config: Config,
                       env: StationaryCheetahWindVelEnv):

        self.config: Config = config
        self.logger: SummaryWriter = config.logger

        self.env: StationaryCheetahWindVelEnv = env
        self.obs_dim: int = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

        self.model: VAE = VAE(config=config,
                              obs_dim=self.obs_dim,
                              action_dim=self.action_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)

        self.train_buffer: VAEBuffer = VAEBuffer(config=config.train_buffer, obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.test_buffer: VAEBuffer = VAEBuffer(config=config.test_buffer, obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.min_loss = np.Inf

    def train_model(self):

        collect_stationary_train_trajectories(self.config, self.env, self.train_buffer)  # stationary trajectories

        for iter_idx in range(self.config.training.pretrain_iter):

            obs, actions, rewards, next_obs = self.train_buffer.sample_batch(batch_size=self.config.training.batch_size)
            state_reconstruction_loss, \
            reward_reconstruction_loss, \
            kl_loss = self.train_iter(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs)

            self.logger.add_scalar(tag='train/state_reconstruction_loss', scalar_value=state_reconstruction_loss, global_step=iter_idx)
            self.logger.add_scalar(tag='train/reward_reconstruction_loss', scalar_value=reward_reconstruction_loss, global_step=iter_idx)
            self.logger.add_scalar(tag='train/kl_loss', scalar_value=kl_loss, global_step=iter_idx)

            if iter_idx % self.config.training.print_train_loss_freq == 0:

                print(f'Train: curr_iter = {iter_idx}, '
                      f'state_loss = {state_reconstruction_loss}, '
                      f'reward_loss = {reward_reconstruction_loss}, '
                      f'kl_loss = {kl_loss}')

            curr_total_loss = state_reconstruction_loss + reward_reconstruction_loss + kl_loss

            if curr_total_loss <= self.min_loss:
                self.min_loss = curr_total_loss
                is_best_result = True
            else:
                is_best_result = False

            if iter_idx % self.config.training.save_freq or is_best_result:

                save_checkpoint(checkpoint_dir=self.logger.log_dir,
                                model=self.model,
                                optimizer=self.optimizer,
                                loss=curr_total_loss,
                                epoch_idx=iter_idx,
                                is_best=is_best_result)

            if iter_idx % self.config.training.eval_freq == 0:

                collect_stationary_test_trajectories(self.config, self.env, self.test_buffer)

                obs, actions, rewards, next_obs = self.test_buffer.sample_batch(batch_size=self.config.test_buffer.max_episode_num)

                state_reconstruction_loss, \
                reward_reconstruction_loss, \
                kl_loss = self.test_iter(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs)

                print(f'Test: curr_iter = {iter_idx}, '
                      f'state_loss = {state_reconstruction_loss}, '
                      f'reward_loss = {reward_reconstruction_loss}, '
                      f'kl_loss = {kl_loss}')

                self.logger.add_scalar('test/state_reconstruction_loss', state_reconstruction_loss, iter_idx)
                self.logger.add_scalar('test/reward_reconstruction_loss', reward_reconstruction_loss, iter_idx)
                self.logger.add_scalar('test/kl_loss', kl_loss, iter_idx)

    def train_iter(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_obs: torch.Tensor):

        self.model.train()

        obs_d = obs.to(self.config.device)
        actions_d = actions.to(self.config.device)
        rewards_d = rewards.to(self.config.device)
        next_obs_d = next_obs.to(self.config.device)

        next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs_d, actions_d, rewards_d, next_obs_d)

        state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs_d)
        reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards_d)
        kl_loss = compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar)

        total_loss = self.config.training.state_reconstruction_loss_weight * state_reconstruction_loss + \
                     self.config.training.reward_reconstruction_loss_weight * reward_reconstruction_loss + \
                     self.config.training.kl_loss_weight * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return state_reconstruction_loss.item(), reward_reconstruction_loss.item(), kl_loss.item()

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

