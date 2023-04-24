import pydantic


class BaseModel(pydantic.BaseModel):

    class Config:
        arbitrary_types_allowed = True
        extra = pydantic.Extra.forbid
    """Disallow extra arguments to init"""