import pydantic


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

class DecoderConfig(BaseModel):

    state = StateDecoderConfig()
    reward = RewardDecoderConfig()