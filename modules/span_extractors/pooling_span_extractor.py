"""
Additional span embedding extractors from
https://github.com/coder318/sodner/blob/master/sodner/models/span_extractors.py
"""

from typing import Callable
from overrides import overrides

import torch

from allennlp.nn import util
from allennlp.common.checks import ConfigurationError
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import (
    SpanExtractorWithSpanWidthEmbedding
)


@SpanExtractor.register("pooling")
class PoolingSpanExtractor(SpanExtractorWithSpanWidthEmbedding):

    def __init__(
        self,
        input_dim: int,
        combination: str = "mean",
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__(
            input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths
        )
        self._combination = combination

        if combination.lower() == 'mean':
            self.combine: Callable = util.masked_mean
        elif combination.lower() == 'max':
            self.combine: Callable = util.masked_max
        else:
            raise ConfigurationError("Unsupport pooling operation: " + combination)

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        if self._span_width_embedding is None:
            return self._input_dim
        return self._input_dim + self._span_width_embedding.embedding_dim

    @overrides
    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> None:
        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = span_indices.split(1, dim=-1)

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_indices_mask_unsqueeze = span_indices_mask.unsqueeze(-1)
            span_starts = span_starts * span_indices_mask_unsqueeze
            span_ends = span_ends * span_indices_mask_unsqueeze

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(
            max_batch_span_width, util.get_device_of(sequence_tensor)
        ).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = max_span_range_indices <= span_widths
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0)
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = self.combine(span_embeddings, span_mask.unsqueeze(-1), dim=2)

        return span_embeddings
