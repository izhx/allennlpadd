import torch
from overrides import overrides

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

from .span_extractor import MySpanExtractor


@SpanExtractor.register("my_self_attentive")
class MySelfAttentiveSpanExtractor(MySpanExtractor):
    """
    Computes span representations by generating an unnormalized attention score for each
    word in the document. Spans representations are computed with respect to these
    scores by normalising the attention scores for words inside the span.
    Given these attention distributions over every span, this module weights the
    corresponding vector representations of the words in the span by this distribution,
    returning a weighted representation of each span.
    Registered as a `SpanExtractor` with name "self_attentive".
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    # Returns
    attended_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is formed by locally normalising a global attention over the sequence. The only way
        in which the attention distribution differs over different spans is in the set of words
        over which they are normalized.
    """

    def __init__(
        self,
        input_dim: int,
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
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.FloatTensor:
        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # shape (batch_size, sequence_length, embedding_dim + 1)
        concat_tensor = torch.cat([sequence_tensor, global_attention_logits], -1)

        concat_output, span_mask = util.batched_span_select(concat_tensor, span_indices)

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = concat_output[:, :, :, :-1]
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_logits = concat_output[:, :, :, -1]

        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_attention_weights = util.masked_softmax(span_attention_logits, span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = util.weighted_sum(span_embeddings, span_attention_weights)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            attended_text_embeddings *= span_indices_mask.unsqueeze(-1)

        if self._span_width_embedding is not None:
            # we combine the span_width_embeddings finally because we may get
            # zero width embeddings for empty spans. We don't want it to be
            # masked. It may be helpful in some tasks.
            attended_text_embeddings = self._combine_span_width_embeddings(
                attended_text_embeddings, span_indices, span_indices_mask
            )

        return attended_text_embeddings
