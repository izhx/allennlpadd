"""
"""

from allennlp.common import Registrable, FromParams
from allennlp.common.params import Params


def construct_registrable(base: Registrable, *args, **kwargs) -> Registrable:
    """
    Instantiating an object of the registered `kwargs[type]` subclass of `base`
    class by `*args`, `**kwargs`. In this way, you can pass extra parameters
    not only from the json config file, but also python code. It is very helpful
    when the output_dim of some module is varies according to configuration and
    need to be passed to other modules.
    """
    cls = base.by_name(kwargs.pop("type"))
    obj = cls(*args, **kwargs)
    return obj


def construct_from_params(cls: FromParams, **kwargs) -> FromParams:
    """
    Just merge additional kwargs. Such as `input_dim=X, **module`.
    """
    return cls.from_params(Params(kwargs))
