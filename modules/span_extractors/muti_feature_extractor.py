"""
A multiple feature SpanExtractor.
"""

from typing import Dict, Optional
from overrides import overrides

import torch

from allennlp.nn import util
from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.span_extractors import (
    SpanExtractor, EndpointSpanExtractor, SelfAttentiveSpanExtractor,
    BidirectionalEndpointSpanExtractor)

# from .conv_extractor import ConvSpanExtractor
from .pooling_extractor import PoolingSpanExtractor


@SpanExtractor.register("multi_feat")
class MultiFeatrueSpanExtractor(SpanExtractor):
    def __init__(
        self,
        input_dim: int,
        endpoint_combination: str = None,
        # conv_combination: str = None,
        pooling_combination: str = None,
        self_attentive: bool = False,
        bidirectional_endpoint: Dict = None,
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
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings, embedding_dim=span_width_embedding_dim
            )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_buckets and span_width_embedding_dim."
            )

        self._output_dim = 0
        self.extractor_forwards = list()

        self._endpoint_extractor: Optional[EndpointSpanExtractor] = None
        if endpoint_combination:
            self._endpoint_extractor = EndpointSpanExtractor(
                input_dim, endpoint_combination)
            self._output_dim += self._endpoint_extractor.get_output_dim()
            self.extractor_forwards.append(self._endpoint_extractor.forward)
        # self._conv_extractor: Optional[ConvSpanExtractor] = None
        # if conv_combination:
        #     self._conv_extractor = ConvSpanExtractor(input_dim, conv_combination)
        #     self._output_dim += self._conv_extractor.get_output_dim()
        #     self.extractor_forwards.append(self._conv_extractor.forward)
        self._pooling_extractor: Optional[PoolingSpanExtractor] = None
        if pooling_combination:
            self._pooling_extractor = PoolingSpanExtractor(input_dim, pooling_combination)
            self._output_dim += self._pooling_extractor.get_output_dim()
            self.extractor_forwards.append(self._pooling_extractor.forward)
        self._self_attentive_extractor: Optional[SelfAttentiveSpanExtractor] = None
        if self_attentive:
            self._self_attentive_extractor = SelfAttentiveSpanExtractor(input_dim)
            self._output_dim += self._self_attentive_extractor.get_output_dim()

            def forward(sequence_tensor, span_indices, _, span_indices_mask):
                return self._self_attentive_extractor(
                    sequence_tensor, span_indices, span_indices_mask
                )
            self.extractor_forwards.append(forward)
        self._bidirectional_endpoint_extractor: Optional[BidirectionalEndpointSpanExtractor] = None
        if bidirectional_endpoint:
            self._bidirectional_endpoint_extractor = BidirectionalEndpointSpanExtractor(
                input_dim, **bidirectional_endpoint)
            self._output_dim += self._bidirectional_endpoint_extractor.get_output_dim()
            self.extractor_forwards.append(self._bidirectional_endpoint_extractor.forward)

        if self._output_dim == 0:
            raise ConfigurationError("No extractor enabled, you must have one")
        if self._span_width_embedding is not None:
            self._output_dim += span_width_embedding_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @overrides
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> None:
        # compute and combine all embeddings
        embeddings = [
            f(sequence_tensor, span_indices, sequence_mask, span_indices_mask)
            for f in self.extractor_forwards
        ]
        combined_tensors = torch.cat(embeddings, -1)

        if self._span_width_embedding is not None:
            # shape (batch_size, num_spans)
            span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

            if span_indices_mask is not None:
                # It's not strictly necessary to multiply the span indices by the mask here,
                # but it's possible that the span representation was padded with something other
                # than 0 (such as -1, which would be an invalid index), so we do so anyway to
                # be safe.
                span_starts = span_starts * span_indices_mask
                span_ends = span_ends * span_indices_mask

            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = util.bucket_values(
                    span_ends - span_starts, num_total_buckets=self._num_width_embeddings  # type: ignore
                )
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            combined_tensors = torch.cat([combined_tensors, span_width_embeddings], -1)

        if span_indices_mask is not None:
            return combined_tensors * span_indices_mask.unsqueeze(-1)

        return combined_tensors
