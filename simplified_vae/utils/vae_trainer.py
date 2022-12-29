import gym
import torch
import time
from torch.utils.tensorboard import SummaryWriter

from simplified_vae.config.config import Config
from simplified_vae.utils.env_utils import sample_trajectory
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, compute_kl_loss
from simplified_vae.models.vae import VAE
from simplified_vae.utils.vae_storage import VAEBuffer


class VAETrainer:

    def __init__(self, config: Config,
                       env: gym.Env):

        self.config: Config = config
        self.logger: SummaryWriter = config.logger

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if discrete else env.action_space.shape[0]

        self.model = VAE(config=config,
                         obs_dim=self.obs_dim,
                         action_dim=self.action_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)

        self.buffer: VAEBuffer = VAEBuffer(config=config, obs_dim=self.obs_dim, action_dim=self.action_dim)

    def train_model(self):

        self.model.train()

        for iter_idx in range(self.config.training.pretrain_iter):

            obs, actions, rewards, next_obs = self.buffer.sample_batch(self.config.training.batch_size)
            state_reconstruction_loss, \
            reward_reconstruction_loss, \
            kl_loss = self.train_iter(obs, actions, rewards, next_obs)

            if iter_idx % 50 == 0:

                print(f'curr_iter = {iter_idx}, '
                      f'state_loss = {state_reconstruction_loss},'
                      f'reward_loss = {reward_reconstruction_loss},'
                      f'kl_loss = {kl_loss}')

            self.logger.add_scalar('loss/state_reconstruction_loss', state_reconstruction_loss, iter_idx)
            self.logger.add_scalar('loss/reward_reconstruction_loss', reward_reconstruction_loss, iter_idx)
            self.logger.add_scalar('loss/kl_loss', kl_loss, iter_idx)

    def train_iter(self, obs: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_obs: torch.Tensor):

        obs_d = obs.to(self.config.device)
        actions_d = actions.to(self.config.device)
        rewards_d = rewards.to(self.config.device)
        next_obs_d = next_obs.to(self.config.device)

        next_obs_preds, rewards_pred, latent_mean, latent_logvar = self.model(obs_d, actions_d, rewards_d, next_obs_d)

        state_reconstruction_loss = compute_state_reconstruction_loss(next_obs_preds, next_obs_d)
        reward_reconstruction_loss = compute_reward_reconstruction_loss(rewards_pred, rewards_d)
        kl_loss = compute_kl_loss(latent_mean=latent_mean, latent_logvar=latent_logvar)

        state_reconstruction_loss = state_reconstruction_loss.sum(dim=-1)
        state_reconstruction_loss = state_reconstruction_loss.sum(dim=-1)
        state_reconstruction_loss = state_reconstruction_loss.mean(dim=0)

        reward_reconstruction_loss = reward_reconstruction_loss.sum(dim=-1)
        reward_reconstruction_loss = reward_reconstruction_loss.sum(dim=-1)
        reward_reconstruction_loss = reward_reconstruction_loss.mean(dim=0)

        kl_loss = kl_loss.sum(dim=-1)
        kl_loss = kl_loss.sum(dim=-1)
        kl_loss = kl_loss.mean(dim=0)

        total_loss = self.config.training.state_reconstruction_loss_weight * state_reconstruction_loss + \
                     self.config.training.reward_reconstruction_loss_weight * reward_reconstruction_loss + \
                     self.config.training.kl_loss_weight * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return state_reconstruction_loss.item(), reward_reconstruction_loss.item(), kl_loss.item()

    def collect_trajectories(self):

        start_time = time.time()

        for trajectory_idx in range(self.config.training.pretrain_collected_epiosde_num):
            obs, actions, rewards, next_obs, dones = sample_trajectory(env=self.env, max_env_steps=100)

            self.buffer.insert(obs=obs,
                               actions=actions,
                               rewards=rewards,
                               next_obs=next_obs,
                               dones=dones)

        stop_time = time.time()
        elapsed_time = stop_time - start_time
        print(f'Took {elapsed_time} seconds to collect {self.config.training.pretrain_collected_epiosde_num} episodes')

