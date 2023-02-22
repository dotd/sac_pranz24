from typing import List, Union
import pydantic
import torch


class BaseModel(pydantic.BaseModel):

    class Config:
        arbitrary_types_allowed = True
        extra = pydantic.Extra.forbid
    """Disallow extra arguments to init"""


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
    use_stationary_trajectories: bool = False

    sum_reward_window_size = 100
    eval_freq: int = 50
    print_train_loss_freq = 50

    save_freq: int = 50


class BufferConfig(BaseModel):

    max_episode_len: int
    max_episode_num: int


class VAETrainBufferConfig(BufferConfig):
    max_episode_len: int = 100
    max_episode_num: int = 5000


class VAETestBufferConfig(BufferConfig):
    max_episode_len: int = 100
    max_episode_num: int = 50

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
    start_steps: int = -1
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
    clusters_num = 10
    cusum_thresh = 7
    meta_dist_num: int = 2
    dist_epsilon = 0.00001

    max_episode_len: int = 100
    max_episode_num: int = 200
    clusters_queue_size: int = 10000
    median_window_size = 20

    cusum_window_length: int = 3000
    env_window_delta = 200
    poisson_freq = 100
    freq_multiplier = 1
    poisson_dist: bool = False

class EnvConfig(BaseModel):
    name: str = 'HalfCheetah-v2'
    max_episode_steps = 100

class StationaryWindvelEnvConfig(EnvConfig):

    name: str = 'HalfCheetah-v2'

    max_episode_steps = 100

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class ToggleWindvelEnvConfig(EnvConfig):

    name: str = 'HalfCheetah-v2'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class FixedToggleWindvelEnvConfig(EnvConfig):

    name: str = 'HalfCheetah-v2'

    max_episode_steps = 1000000

    low_target_vel: float = 0.
    high_target_vel: float = 3.
    low_wind_frc: float = 0.
    high_wind_frc: float = 20.


class StationaryABSEnvConfig(EnvConfig):

    """ Constructor.
     Parameters
     ----------
     cp_brake: float
         Brake pressure coefficient [n/a].
     r_wheel: float
         Wheel radius [$m$].
     j_wheel: float
         Wheel moment of inertia [$kg m^2$].
     fn_vehicle: float
         Normal force (i.e. downward force from car weight) [$N$].
     vx_vehicle: float
         Vehicle speed in x direction [$m/s$].
     lp_filter_1: float
         Hydraulic low-pass filter coefficient [n/a].
     lp_filter_2: float
         Hydraulic low-pass filter coefficient [n/a]
     tire_b: float
         Magic-formula tire B coefficient [n/a].
     tire_c: float
         Magic-formula tire C coefficient [n/a].
     tire_d: float
         Magic-formula tire D coefficient [n/a].
     tire_e: float
         Magic-formula tire E coefficient [n/a].
     dt_sim: float
         Simulation discretization [s].
     T_sim: float
         Simulation time horizon [s].
     """

    name: str = 'FixedABS'

    cp_brake: float = 43.
    r_wheel: float = 0.3657
    j_wheel: float = 2.3120
    fn_vehicle: float = 1e4
    vx_vehicle: float = 30.
    lp_filter_1: float = 0.02
    lp_filter_2: float = 0.066
    tire_b: float = 12  # 32.609
    tire_c: float = 2.3  # 1.533
    tire_d: float = 1.  # 1.3
    tire_e: float = 0.97  # 0.8
    dt_sim: float = 1e-3
    T_sim: float = 1.0
    max_episode_steps = 500

    # Spaces
    observation_lims_low: List = [0., 0., 0.]
    obsevation_lims_high: List = [100., 300., 300.]

    action_lims_low = 0.
    action_lims_high = 300.


class ToggleABSEnvConfig(EnvConfig):

    name: str = 'ToggleABS'

    cp_brake: float = 43.
    r_wheel: float = 0.3657
    j_wheel: float = 2.3120
    fn_vehicle: float = 1e4
    vx_vehicle: float = 30.
    lp_filter_1: float = 0.02
    lp_filter_2: float = 0.066
    tire_b: float = 12  # 32.609
    tire_c: float = 2.3  # 1.533
    tire_d: float = 1.  # 1.3
    tire_e: float = 0.97  # 0.8
    dt_sim: float = 1e-3
    T_sim: float = 1.0
    max_episode_steps = 500


class FixedToggleABSEnvConfig(EnvConfig):

    name: str = 'ToggleABS'

    cp_brake: float = 43.
    r_wheel: float = 0.3657
    j_wheel: float = 2.3120
    fn_vehicle: float = 1e4
    vx_vehicle: float = 30.
    lp_filter_1: float = 0.02
    lp_filter_2: float = 0.066
    tire_b: float = 12  # 32.609
    tire_c: float = 2.3  # 1.533
    tire_d: float = 1.  # 1.3
    tire_e: float = 0.97  # 0.8
    dt_sim: float = 1e-3
    T_sim: float = 1.0
    max_episode_steps = 500


class BaseConfig(BaseModel):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed: int = 0

    agent: AgentConfig = AgentConfig()
    cpd: CPDConfig = CPDConfig()
    model: ModelConfig = ModelConfig()
    training: VAETrainingConfig = VAETrainingConfig()
    env: EnvConfig
    train_buffer: TrainBufferConfig = TrainBufferConfig()
    test_buffer: TestBufferConfig = TestBufferConfig()
    vae_train_buffer: VAETrainBufferConfig = VAETrainBufferConfig()
    vae_test_buffer: VAETestBufferConfig = VAETestBufferConfig()

