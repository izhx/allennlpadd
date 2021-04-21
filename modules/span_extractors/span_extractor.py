"""
Fixed span extractor.
"""

from typing import Optional

import torch
from torch.nn.modules import Embedding

from allennlp.modules.span_extractors import SpanExtractor
from allennlp.nn import util
from allennlp.common.checks import ConfigurationError


class MySpanExtractor(SpanExtractor):
    """
    Many NLP models deal with representations of spans inside a sentence.
    SpanExtractors define methods for extracting and representing spans
    from a sentence.

    SpanExtractors take a sequence tensor of shape (batch_size, timesteps, embedding_dim)
    and indices of shape (batch_size, num_spans, 2) and return a tensor of
    shape (batch_size, num_spans, ...), forming some representation of the
    spans.

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    """

    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._span_width_embedding: Optional[Embedding] = None
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            # we use pytorch embedding because it is enough.
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings, embedding_dim=span_width_embedding_dim
            )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )

    def _combine_span_width_embeddings(
        self,
        span_embeddings: torch.Tensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Embed span width to vectors and combine them with `span_embeddings`.
        It will give a learned zero width embedding for the empty span
        if `span_indices_mask` is given.
        """
        if self._span_width_embedding is None:
            raise RuntimeError("Can not embed span width")

        # width = end_indexs - start_index + 1 since `SpanField` use inclusive indices.
        # shape (batch_size, num_spans)
        span_widths = span_indices[..., 1] - span_indices[..., 0] + 1

        if span_indices_mask is not None:
            # set width of empty spans to zero, so that they can be embeded correctly.
            span_widths = span_widths * span_indices_mask

        if self._bucket_widths:
            # convert span widths to bucketed values
            span_widths = util.bucket_values(
                span_widths, num_total_buckets=self._num_width_embeddings  # type: ignore
            )

        max_span_width = span_widths.max().item()
        if max_span_width >= self._num_width_embeddings:
            # In this case, the indices has exceeded the embedding matrix
            raise RuntimeError(
                "The span width {} is out of width embeddings range 0-{}".format(
                    max_span_width, self._num_width_embeddings - 1
                )
            )

        # Embed the span widths and concatenate to the rest of the representations.
        span_width_embeddings = self._span_width_embedding(span_widths)
        combined_tensors = torch.cat([span_embeddings, span_width_embeddings], -1)

        return combined_tensors
