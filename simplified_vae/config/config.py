from typing import List

import numpy as np
import pydantic
import torch
import datetime

from torch.utils.tensorboard import SummaryWriter


class BaseModel(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    """Disallow extra arguments to init"""
    pass


class EncoderConfig(BaseModel):

    action_embed_dim = 16
    obs_embed_dim = 32
    reward_embed_dim = 16

    additional_embed_layers: List = [64, 32, 16]
    recurrent_hidden_dim = 128
    vae_hidden_dim = 5


class StateDecoderConfig(BaseModel):

    layers = [64, 32]


class RewardDecoderConfig(BaseModel):

    layers = [64, 32]


class TaskDecoderConfig(BaseModel):

    layers = [64, 32]


class TrainingConfig(BaseModel):

    lr: float = 0.001
    batch_size: int = 32
    state_reconstruction_loss_weight: float = 1.0
    reward_reconstruction_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    pretrain_iter = 100000

    use_kl_posterior_loss: bool = False
    use_stationary_trajectories: bool = False

    env_change_freq: int = 100
    eval_freq: int = 50
    print_train_loss_freq = 50

    save_freq: int = 50


class TaskConfig(BaseModel):

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class BufferConfig(BaseModel):

    max_episode_len: int
    max_episode_num: int


class TrainBufferConfig(BufferConfig):

    max_episode_len: int = 100
    max_episode_num: int = 5000


class TestBufferConfig(BufferConfig):

    max_episode_len: int = 100
    max_episode_num: int = 5000


class ModelConfig(BaseModel):

    use_rnn_model: bool = False

    encoder: EncoderConfig = EncoderConfig()
    state_decoder: StateDecoderConfig = StateDecoderConfig()
    reward_decoder: RewardDecoderConfig = RewardDecoderConfig()
    task_decoder: TaskDecoderConfig = TaskDecoderConfig()


class ClusteringConfig(BaseModel):

    clusters_num = 50


class TWRConfig(BaseModel):

    obs_shape: int = 10
    latents_shape: int = 5
    hidden_size: int = 32
    n_layers: int = 5
    init_std: int = - 1
    loss_type: str = 'KL' # 'KL', 'exp' or 'sqrt'

    n_trajectories: int = 1
    length: int = 1000
    change_point: int = 500

    t_init: int = 50
    t_end: int = 1000
    n_epochs: int = 30
    batch_size: int = 32
    epsilon: float = .01 # speed of annealing
    t_min: int = 10
    annealing: bool = True

    annealing_coef: float = .01
    lr: float = 0.001
    train_eps: float = 1e-7
    seed: int = 0

class Config:

    env_name: str = 'HalfCheetah-v3'
    device: int = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed: int = 0

    logger = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_VAE')

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    task: TaskConfig = TaskConfig()
    train_buffer: TrainBufferConfig = TrainBufferConfig()
    test_buffer: TestBufferConfig = TestBufferConfig()

    clustering: ClusteringConfig = ClusteringConfig()
    twr: TWRConfig = TWRConfig()
