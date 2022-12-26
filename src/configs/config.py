import pydantic


class BaseModel(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    """Disallow extra arguments to init"""
    pass


class PolicyConfig:

    policy_type: str = 'Gaussian'  # Gaussian | Deterministic
    gamma: float = 0.99 # Discount factor for reward
    tau: float = 0.005 # target smoothing coefficient(τ)
    lr: float = 0.0003 # learning rate
    alpha: float = 0.2 # Temperature parameter α determines the relative importance of the entropy
    automatic_entropy_tuning: bool = False # Automaically adjust α

    seed: int = 123456


class TrainingConfig:

    batch_size: int = 256
    max_num_steps: int = 1000001
    hidden_size: int = 256
    updates_per_step: int = 1
    start_steps: int = 10000 # Steps sampling random actions
    target_update_interval: int = 1 # Value target update per no. of updates per step
    replay_size: int = 1000000
    cuda: bool = True
    save_episodes: int = 2


class BaseConfig(BaseModel):

    env_name: str = 'HalfCheetah-v3'
    training = TrainingConfig()
    policy = PolicyConfig()

