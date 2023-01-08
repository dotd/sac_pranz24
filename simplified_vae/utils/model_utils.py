from typing import List

import torch

from simplified_vae.config.config import Config
from simplified_vae.models.vae import RNNVAE, VAE
from simplified_vae.utils.logging_utils import load_checkpoint


def init_model(config: Config,
               obs_dim: int,
               action_dim: int):

    model = None

    if config.model.type == 'RNNVAE':
        model = RNNVAE(config=config,
                       obs_dim=obs_dim,
                       action_dim=action_dim)

    elif config.model.type == 'VAE':
        model = VAE(config=config,
                    obs_dim=obs_dim,
                    action_dim=action_dim)

    elif config.model.type == 'AE':
        raise NotImplementedError

    else:
        raise NotImplementedError

    model, epoch, loss = load_checkpoint(checkpoint_path=config.model.checkpoint_path, model=model, optimizer=None)

    return model, epoch, loss


def all_to_device(*args, device = None):

    args_d = []
    for arg in args:
        args_d.append(arg.to(device))

    return tuple(args_d)


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

        return all_latent_meanss