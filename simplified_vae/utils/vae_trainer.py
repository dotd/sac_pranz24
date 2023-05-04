from typing import Union, List

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import BaseConfig
from simplified_vae.env.stationary_abs_env import StationaryABSEnv
from simplified_vae.utils.env_utils import collect_stationary_trajectories, collect_non_stationary_trajectories
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelWrapper
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, \
    compute_kl_loss, compute_kl_loss_with_posterior
from simplified_vae.models.vae import VAE, RNNVAE
from simplified_vae.utils.vae_storage import Buffer
from simplified_vae.utils.logging_utils import save_checkpoint, write_config


class VAETrainer:

    def __init__(self,
                 config: BaseConfig,
                 env: StationaryCheetahWindVelWrapper,
                 logger: SummaryWriter):

        self.config: BaseConfig = config
        self.logger: SummaryWriter = logger

        self.env: Union[StationaryCheetahWindVelWrapper, StationaryABSEnv] = env
        self.obs_dim: int = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

        self.rg = np.random.RandomState(seed=self.config.seed)

        if self.config.model.type == 'RNNVAE':
            self.model: RNNVAE = RNNVAE(config=config,
                                        obs_dim=self.obs_dim,
                                        action_dim=self.action_dim)

        elif self.config.model.type == 'VAE':
            self.model: VAE = VAE(config=config,
                                  obs_dim=self.obs_dim,
                                  action_dim=self.action_dim)
        else:
            raise NotImplementedError

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)

        self.train_buffer: Buffer = Buffer(max_total_steps=self.config.vae_train_buffer.max_total_steps,
                                           obs_dim=self.obs_dim,
                                           action_dim=self.action_dim)

        self.test_buffer: Buffer = Buffer(max_total_steps=self.config.vae_test_buffer.max_total_steps,
                                          obs_dim=self.obs_dim,
                                          action_dim=self.action_dim)
        self.min_loss = np.Inf

        write_config(config=config, logdir=self.logger.log_dir)

    def train_model(self):

        if self.config.training.use_stationary_trajectories:
            collect_stationary_trajectories(env=self.env,
                                            buffer=self.train_buffer,
                                            max_env_steps=self.config.vae_train_buffer.max_env_steps,
                                            max_total_steps=self.config.vae_train_buffer.max_total_steps,
                                            env_change_freq=self.config.vae_train_buffer.max_total_steps // 10,
                                            is_print=True)
        else:
            collect_non_stationary_trajectories(env=self.env,
                                                buffer=self.train_buffer,
                                                max_env_steps=self.config.vae_train_buffer.max_env_steps, # TODO fix this function
                                                max_total_steps=self.config.vae_train_buffer.max_episode_num,
                                                env_change_freq=self.config.vae_train_buffer.max_episode_len,
                                                rg=self.rg,
                                                is_print=True)

        for iter_idx in range(self.config.training.vae_train_iter):

            obs, actions, rewards, next_obs, lengths = self.train_buffer.sample_batch(batch_size=self.config.training.batch_size)
            state_reconstruction_loss, \
            reward_reconstruction_loss, \
            kl_loss = self.train_iter(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, lengths=lengths)

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

                if self.config.training.use_stationary_trajectories:

                    collect_stationary_trajectories(env=self.env,
                                                    buffer=self.test_buffer,
                                                    max_env_steps=self.config.vae_test_buffer.max_env_steps,
                                                    max_total_steps=self.config.vae_test_buffer.max_total_steps,
                                                    env_change_freq=1,
                                                    is_print=False)
                else:
                    collect_non_stationary_trajectories(env=self.env,
                                                        buffer=self.test_buffer,
                                                        episode_num=self.config.vae_test_buffer.max_episode_num, # TODO fix
                                                        episode_len=self.config.vae_test_buffer.max_episode_len,
                                                        rg=self.rg,
                                                        is_print=False)

                obs, actions, rewards, next_obs, lengths = self.test_buffer.sample_batch(batch_size=self.config.test_buffer.max_episode_num)

                state_reconstruction_loss, \
                reward_reconstruction_loss, \
                kl_loss = self.test_iter(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, lengths=lengths)

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
                         next_obs: torch.Tensor,
                         lengths: List):

        self.model.train()

        obs_d = obs.to(self.config.device)
        actions_d = actions.to(self.config.device)
        rewards_d = rewards.to(self.config.device)
        next_obs_d = next_obs.to(self.config.device)

        if self.config.model.type == 'RNNVAE':
            next_obs_preds, rewards_pred, latent_mean, latent_logvar, _, _ = self.model(obs=obs_d, actions=actions_d, rewards=rewards_d, next_obs=next_obs_d, lengths=lengths)
        elif self.config.model.type == 'VAE':
            next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs=obs_d, actions=actions_d, rewards=rewards_d, next_obs=next_obs_d)
        else:
            raise NotImplementedError

        # TODO next_obs_preds outputs for the padded parts should be zero
        state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs_d)
        reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards_d)

        if self.config.training.use_kl_posterior_loss:
            kl_loss = compute_kl_loss_with_posterior(latent_mean=latent_mean, latent_logvar=latent_logvar)
        else:
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
                        next_obs: torch.Tensor,
                        lengths: List):

        self.model.eval()

        with torch.no_grad():

            obs_d = obs.to(self.config.device)
            actions_d = actions.to(self.config.device)
            rewards_d = rewards.to(self.config.device)
            next_obs_d = next_obs.to(self.config.device)

            if self.config.model.type == 'RNNVAE':
                next_obs_preds, rewards_pred, latent_mean, latent_logvar, _, _ = self.model(obs=obs_d, actions=actions_d, rewards=rewards_d, next_obs=next_obs_d, lengths=lengths)
            elif self.config.model.type == 'VAE':
                next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs=obs_d, actions=actions_d, rewards=rewards_d, next_obs=next_obs_d)
            else:
                raise NotImplementedError

            state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs_d)
            reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards_d)
            kl_loss = compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar)

            return state_reconstruction_loss.item(), reward_reconstruction_loss.item(), kl_loss.item()

