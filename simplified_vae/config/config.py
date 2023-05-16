from typing import List
import torch

from simplified_vae.config.envs_config import EnvConfig
from simplified_vae.config.pydantic_config import BaseModel


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


class VAETrainingConfig(BaseModel):

    lr: float = 0.001
    batch_size: int = 32
    state_reconstruction_loss_weight: float = 1.0
    reward_reconstruction_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    vae_train_iter = 100000

    use_kl_posterior_loss: bool = False
    use_stationary_trajectories: bool = True

    sum_reward_window_size = 10
    eval_freq: int = 50
    print_train_loss_freq = 10

    save_freq: int = 50


class BufferConfig(BaseModel):

    max_episode_len: int
    max_episode_num: int


class VAETrainBufferConfig(BaseModel):

    max_env_steps: int = 500
    max_total_steps: int = 200000


class VAETestBufferConfig(BaseModel):

    max_env_steps: int = 500
    max_total_steps: int = 20000


class TrainBufferConfig(BufferConfig):

    max_episode_len: int = 1000000
    max_episode_num: int = 2


class TestBufferConfig(BufferConfig):

    max_episode_len: int = 100
    max_episode_num: int = 5000


class AgentConfig(BaseModel):

    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 0.0003
    alpha: float = 0.2
    automatic_entropy_tuning: bool = False
    batch_size: int = 256
    num_steps: int = 1000001
    hidden_size: int = 256
    updates_per_step: int = 1
    start_steps: int = 10000
    target_update_interval: int = 1
    replay_size: int = 1000000

    policy_type: str = 'Gaussian'
    target_update_interval = target_update_interval
    automatic_entropy_tuning = automatic_entropy_tuning

    agents_num: int = 2


class ModelConfig(BaseModel):

    use_rnn_model: bool = False
    type: str = 'RNNVAE'
    checkpoint_path = 'runs/2023-01-02_09-12-57_VAE/model_best.pth.tar'

    encoder: EncoderConfig = EncoderConfig()
    state_decoder: StateDecoderConfig = StateDecoderConfig()
    reward_decoder: RewardDecoderConfig = RewardDecoderConfig()
    task_decoder: TaskDecoderConfig = TaskDecoderConfig()


class CPDConfig(BaseModel):

    alpha_val: float = 0.5
    clusters_num = [5, 5, 5] # obs, action, next_obs
    cusum_thresh = 100
    meta_dist_num: int = 2
    dist_epsilon = 0.00001

    max_env_steps: int = 500
    max_total_steps = 1000
    clusters_queue_size: int = 2000
    median_window_size = 20

    cusum_window_length: int = 1000
    env_window_delta = 500 # change to support small trajectories
    poisson_freq = 100
    freq_multiplier = 1
    poisson_dist: bool = False

    transition_cusum_eps: float = 0.1
    prior_cusum_eps: float = 0.01


class BaseConfig(BaseModel):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed: int = 1

    agent: AgentConfig = AgentConfig()
    cpd: CPDConfig = CPDConfig()
    model: ModelConfig = ModelConfig()
    training: VAETrainingConfig = VAETrainingConfig()
    env: EnvConfig = EnvConfig()
    train_buffer: TrainBufferConfig = TrainBufferConfig()
    test_buffer: TestBufferConfig = TestBufferConfig()
    vae_train_buffer: VAETrainBufferConfig = VAETrainBufferConfig()
    vae_test_buffer: VAETestBufferConfig = VAETestBufferConfig()

