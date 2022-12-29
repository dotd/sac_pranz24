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

    recurrent_hidden_dim = 128
    vae_hidden_dim = 5


class StateDecoderConfig(BaseModel):

    layers = [64, 32]


class RewardDecoderConfig(BaseModel):

    layers = [64, 32]


class TrainingConfig(BaseModel):

    lr: float = 0.001
    batch_size: int = 32
    state_reconstruction_loss_weight: float = 1.0
    reward_reconstruction_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    pretrain_iter = 100000
    pretrain_collected_epiosde_num = 1000


class BufferConfig(BaseModel):

    max_episode_len: int = 100
    max_episode_num: int = 10000


class Config:

    env_name: str = 'HalfCheetah-v3'
    device: int = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed: int = 0

    logger = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_VAE')

    buffer: BufferConfig = BufferConfig()
    training: TrainingConfig = TrainingConfig()
    encoder: EncoderConfig = EncoderConfig()
    state_decoder: StateDecoderConfig = StateDecoderConfig()
    reward_decoder: RewardDecoderConfig = RewardDecoderConfig()
