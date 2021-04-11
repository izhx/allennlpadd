"""
"""

from allennlp.common import Registrable


def construct_registrable(base: Registrable, *args, **kwargs):
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
