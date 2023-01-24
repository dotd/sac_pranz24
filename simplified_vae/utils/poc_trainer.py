import itertools
from collections import deque
from typing import Union

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from simplified_vae.config.config import Config
from simplified_vae.env.toggle_windvel_env import ToggleWindVelEnv
from simplified_vae.utils.clustering_utils import Clusterer
from simplified_vae.utils.cpd_utils import CPD
from simplified_vae.utils.env_utils import collect_stationary_trajectories, collect_non_stationary_trajectories, \
    collect_toggle_trajectories
from simplified_vae.env.stationary_cheetah_windvel_wrapper import StationaryCheetahWindVelEnv
from simplified_vae.utils.losses import compute_state_reconstruction_loss, compute_reward_reconstruction_loss, \
    compute_kl_loss
from simplified_vae.utils.model_utils import init_model, all_to_device
from simplified_vae.utils.vae_storage import Buffer
from simplified_vae.utils.logging_utils import write_config
from simplified_vae.models.sac import SAC
from src.utils.replay_memory import ReplayMemory


class POCTrainer:

    def __init__(self, config: Config,
                       env: Union[StationaryCheetahWindVelEnv, ToggleWindVelEnv],
                       data_collection_env: Union[StationaryCheetahWindVelEnv, ToggleWindVelEnv]):

        self.config: Config = config
        self.logger: SummaryWriter = config.logger

        self.env: Union[StationaryCheetahWindVelEnv, ToggleWindVelEnv] = env
        self.data_collection_env: StationaryCheetahWindVelEnv = data_collection_env

        self.obs_dim: int = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim: int = env.action_space.n if discrete else env.action_space.shape[0]

        self.rg = np.random.RandomState(seed=self.config.seed)

        self.model, epoch, loss = init_model(config=config,
                                             obs_dim=self.obs_dim,
                                             action_dim=self.action_dim)

        self.agents = [SAC(config=config,
                           num_inputs=self.obs_dim,
                           action_space=env.action_space) for _ in range(config.agent.agents_num)]

        cpd_num = self.config.cpd.cusum_window_length
        self.cpd = CPD(config=self.config, window_length=self.config.cpd.cusum_window_length)

        self.clusterer = Clusterer(config=self.config, rg=self.rg)

        # Init Buffer
        self.buffer = Buffer(max_episode_num=config.cpd.max_episode_num,
                             max_episode_len=config.cpd.max_episode_len,
                             obs_dim=self.obs_dim,
                             action_dim=self.action_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.lr)
        write_config(config=config, logdir=self.logger.log_dir)

    def init_clusters(self):

        # Collect episodes from Task_0
        self.data_collection_env.set_task(task=self.env.tasks[0])
        collect_stationary_trajectories(env=self.data_collection_env,
                                        buffer=self.buffer,
                                        episode_num=self.config.cpd.max_episode_num // 2,
                                        episode_len=self.config.cpd.max_episode_len,
                                        env_change_freq=self.config.cpd.max_episode_num,
                                        agent=self.agents[0])

        # collect episode from Task_1
        self.data_collection_env.set_task(task=self.env.tasks[1])
        collect_stationary_trajectories(env=self.data_collection_env,
                                        buffer=self.buffer,
                                        episode_num=self.config.cpd.max_episode_num // 2,
                                        episode_len=self.config.cpd.max_episode_len,
                                        env_change_freq=self.config.cpd.max_episode_num,
                                        agent=self.agents[1])

        obs_0, actions_0, rewards_0, next_obs_0 = self.buffer.sample_section(start_idx=0,
                                                                             end_idx=self.config.cpd.max_episode_num // 2)
        obs_1, actions_1, rewards_1, next_obs_1 = self.buffer.sample_section(start_idx=self.config.cpd.max_episode_num // 2,
                                                                             end_idx=self.config.cpd.max_episode_num)

        obs_0_d, actions_0_d, rewards_0_d, next_obs_0 = all_to_device(obs_0, actions_0, rewards_0, next_obs_0, device=self.config.device)
        obs_1_d, actions_1_d, rewards_1_d, next_obs_1 = all_to_device(obs_1, actions_1, rewards_1, next_obs_1, device=self.config.device)

        self.model.eval()
        with torch.no_grad():
            latent_sample_0, latent_mean_0, latent_logvar_0, output_0, hidden_0 = self.model.encoder(obs=obs_0_d,
                                                                                                     actions=actions_0_d,
                                                                                                     rewards=rewards_0_d)

            latent_sample_1, latent_mean_1, latent_logvar_1, output_1, hidden_1 = self.model.encoder(obs=obs_1_d,
                                                                                                     actions=actions_1_d,
                                                                                                     rewards=rewards_1_d)

        latent_mean = torch.cat([latent_mean_0, latent_mean_1], dim=0)
        latent_mean_h = latent_mean.detach().cpu().numpy()

        all_labels = self.clusterer.init_clusters(latent_mean_h)

        task_num = len(self.env.tasks)
        per_task_sample_num = self.config.cpd.max_episode_len * self.config.cpd.max_episode_num // task_num

        self.cpd.dists[0].init_transitions(labels=all_labels[:per_task_sample_num])
        self.cpd.dists[1].init_transitions(labels=all_labels[per_task_sample_num:])

    def train_model(self):

        total_steps = 0
        updates = 0
        hidden_state = None

        sum_reward_window = deque(maxlen=self.config.training.sum_reward_window_size)

        for i_episode in itertools.count(1):

            episode_reward = 0
            episode_steps = 0
            done = False

            self.env.set_task(self.env.tasks[0])
            obs = self.env.reset()

            prev_label = None
            curr_agent_idx = 0

            while not done:

                curr_agent = self.agents[curr_agent_idx]

                if self.config.agent.start_steps > total_steps:
                    action = self.env.action_space.sample()  # Sample random action
                else:
                    action = curr_agent.select_action(obs)  # Sample action from policy

                if len(curr_agent.replay_memory) > self.config.agent.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.config.agent.updates_per_step):

                        # Update parameters of all the networks
                        critic_1_loss, \
                        critic_2_loss, \
                        policy_loss, \
                        ent_loss, \
                        alpha = curr_agent.update_parameters(memory=curr_agent.replay_memory,
                                                             batch_size=self.config.agent.batch_size,
                                                             updates=updates)

                        self.logger.add_scalar('agent_loss/critic_1', critic_1_loss, updates)
                        self.logger.add_scalar('agent_loss/critic_2', critic_2_loss, updates)
                        self.logger.add_scalar('agent_loss/policy', policy_loss, updates)
                        self.logger.add_scalar('agent_loss/entropy_loss', ent_loss, updates)
                        self.logger.add_scalar('entropy_temprature/alpha', alpha, updates)
                        wandb.log({'agent_loss/critic_1': critic_1_loss, 'agent_loss/critic_2': critic_2_loss,
                                   'agent_loss/policy': policy_loss, 'agent_loss/entropy_loss': ent_loss,
                                   'entropy_temprature/alpha': alpha}, step=updates)

                        updates += 1

                next_obs, reward, done, _ = self.env.step(action)  # Step

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == self.env._max_episode_steps else float(not done)
                curr_agent.replay_memory.push(obs, action, reward, next_obs, mask)  # Append transition to memory

                # update CPD estimation
                hidden_state, curr_label, n_c, g_k = self.update_cpd(obs=obs,
                                                                     action=action,
                                                                     reward=reward,
                                                                     hidden_state=hidden_state,
                                                                     prev_label=prev_label,
                                                                     episode_steps=episode_steps,
                                                                     curr_agent_idx=curr_agent_idx)

                if n_c:  # change has been detected
                    curr_agent_idx = int(not curr_agent_idx)


                else:  # no change, update current transition matrix
                    self.agents[curr_agent_idx].transition_mat = self.cpd.dists[curr_agent_idx].transition_mat / \
                                                                 self.cpd.dists[curr_agent_idx].column_sum_vec

                # Update policy if CPD is detected
                # curr_agent_idx = self.update_policy(n_c, curr_agent_idx, episode_steps=episode_steps)

                obs = next_obs
                prev_label = curr_label
                episode_steps += 1
                total_steps += 1
                episode_reward += reward

                sum_reward_window.append(reward)
                if total_steps > self.config.training.sum_reward_window_size:
                    sum_reward = sum([curr for curr in sum_reward_window])
                    self.logger.add_scalar('reward/train', sum_reward, total_steps)
                    wandb.log({'reward/train': sum_reward}, step=total_steps)

                    if total_steps % self.config.training.print_train_loss_freq == 0:
                        print(f'Curr Idx = {total_steps}, Sum Reward = {sum_reward}')

            if total_steps > self.config.agent.num_steps:
                break

            # self.logger.add_scalar('reward/train', episode_reward, i_episode)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_steps,
                                                                                          episode_steps,
                                                                                          round(episode_reward, 2)))

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
            # print(f'curr label = {curr_label}')
            if episode_steps > 0:
                n_c, g_k = self.cpd.update_transition((prev_label.item(), curr_label.item()))
                if n_c:
                    self.add_meta_distribution(self.cpd)

                # curr_cpd_estim = [cpd.update_transition((prev_label, curr_label)) for cpd in self.cpds]

            if episode_steps == 80:
                a = 1
            done = done or (episode_steps >= self.config.train_buffer.max_episode_len)
            obs = next_obs
            prev_label = curr_label
            episode_steps += 1

    def update_cpd(self,
                   obs: torch.Tensor,
                   action: torch.Tensor,
                   reward: torch.Tensor,
                   hidden_state: torch.Tensor,
                   prev_label: np.ndarray,
                   episode_steps: int,
                   curr_agent_idx: int):

        self.model.eval()
        with torch.no_grad():
            # perform add_transition and cusum
            curr_latent_sample, \
            curr_latent_mean, \
            curr_latent_logvar, \
            curr_output, hidden_state = self.model.encoder(obs=obs,
                                                           actions=action,
                                                           rewards=np.array([reward]),
                                                           hidden_state=hidden_state)

        self.clusterer.update_clusters(new_obs=curr_latent_mean)
        curr_label = self.clusterer.predict(curr_latent_mean)

        # print(f'curr label = {curr_label}')
        if episode_steps > 0:
            n_c, g_k = self.cpd.update_transition(curr_transition=(prev_label.item(), curr_label.item()),
                                                     curr_agent_idx=curr_agent_idx)
        else:
            n_c, g_k = None, None

        return hidden_state, curr_label, n_c, g_k

    def update_policy(self, n_c, curr_agent_idx: int, episode_steps: int):

        if n_c: # change has been detected
            curr_agent_idx = int(not curr_agent_idx)

        else: # no change, update current transition matrix
            self.agents[curr_agent_idx].transition_mat = self.cpd.dists[curr_agent_idx].transition_mat / \
                                                         self.cpd.dists[curr_agent_idx].column_sum_vec

        return curr_agent_idx

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

