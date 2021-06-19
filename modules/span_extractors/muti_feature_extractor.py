"""
A multiple feature SpanExtractor.
"""

from typing import Dict, Optional
from overrides import overrides

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.span_extractors import (
    SpanExtractor, EndpointSpanExtractor, SelfAttentiveSpanExtractor,
    BidirectionalEndpointSpanExtractor)
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import (
    SpanExtractorWithSpanWidthEmbedding
)

from .pooling_span_extractor import PoolingSpanExtractor


@SpanExtractor.register("multi_feat")
class MultiFeatrueSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    def __init__(
        self,
        input_dim: int,
        endpoint_combination: str = None,
        pooling_combination: str = None,
        self_attentive: bool = False,
        bidirectional_endpoint: Dict = None,
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
        self._output_dim = 0
        self.extractor_forwards = list()

        self._endpoint_extractor: Optional[EndpointSpanExtractor] = None
        if endpoint_combination:
            self._endpoint_extractor = EndpointSpanExtractor(
                input_dim, endpoint_combination)
            self._output_dim += self._endpoint_extractor.get_output_dim()
            self.extractor_forwards.append(self._endpoint_extractor.forward)
        self._pooling_extractor: Optional[PoolingSpanExtractor] = None
        if pooling_combination:
            self._pooling_extractor = PoolingSpanExtractor(input_dim, pooling_combination)
            self._output_dim += self._pooling_extractor.get_output_dim()
            self.extractor_forwards.append(self._pooling_extractor.forward)
        self._self_attentive_extractor: Optional[SelfAttentiveSpanExtractor] = None
        if self_attentive:
            self._self_attentive_extractor = SelfAttentiveSpanExtractor(input_dim)
            self._output_dim += self._self_attentive_extractor.get_output_dim()
            self.extractor_forwards.append(self._self_attentive_extractor.forward)
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
    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> None:
        # compute and combine all kinds of span embeddings
        combined_tensors = torch.cat([
            f(sequence_tensor, span_indices, sequence_mask, span_indices_mask)
            for f in self.extractor_forwards
        ], dim=-1)
        return combined_tensors
