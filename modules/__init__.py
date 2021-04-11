"""
Custom PyTorch
`Module <https://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP `Model` s.
"""

from .span_extractors import MultiFeatrueSpanExtractor, PoolingSpanExtractor
from .token_embedders import (
    AdapterTransformerEmbedder, AdapterTransformerMismatchedEmbedder)
