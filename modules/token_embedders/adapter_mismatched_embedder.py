"""
Adapter for mismatched transformers in AllenNLP.
"""

from typing import Optional, Dict, Any, Union, List

from overrides import overrides
import torch

from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn import util

from .adapter_embedder import AdapterTransformerEmbedder


@TokenEmbedder.register("adapter_transformer_mismatched")
class AdapterTransformerMismatchedEmbedder(TokenEmbedder):
    """
    The adapter version of `PretrainedTransformerMismatchedEmbedder`.
    Just replaced `self._matched_embedder`.
    """

    def __init__(
        self,
        model_name: str,
        adapter_size: int = 64,
        adapter_num: int = 12,
        external_param: Union[bool, List[bool]] = False,
        max_length: int = None,
        last_layer_only: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = AdapterTransformerEmbedder(
            model_name,
            adapter_size=adapter_size,
            adapter_num=adapter_num,
            external_param=external_param,
            max_length=max_length,
            last_layer_only=last_layer_only,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        return orig_embeddings
