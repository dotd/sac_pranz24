from typing import List

import torch

from simplified_vae.config.config import Config
from simplified_vae.models.vae import RNNVAE, VAE
from simplified_vae.utils.logging_utils import load_checkpoint


def init_model(model_type: str,
               checkpoint_path: str,
               config: Config,
               obs_dim: int,
               action_dim: int):

    model = None

    if model_type == 'RNNVAE':
        model = RNNVAE(config=config,
                       obs_dim=obs_dim,
                       action_dim=action_dim)
    elif model_type == 'VAE':
        model = VAE(config=config,
                    obs_dim=obs_dim,
                    action_dim=action_dim)

    elif model_type == 'AE':
        raise NotImplementedError

    else:
        raise NotImplementedError

    model, epoch, loss = load_checkpoint(checkpoint_path=checkpoint_path, model=model, optimizer=None)

    return model, epoch, loss


def all_to_device(*args, device = None):

    args_d = []
    for arg in args:
        args_d.append(arg.to(device))

    return tuple(args_d)
