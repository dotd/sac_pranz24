import pydantic


class BaseModel(pydantic.BaseModel, extra=pydantic.Extra.forbid):
    """Disallow extra arguments to init"""
    pass

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