"""
Assorted utilities for working with neural networks.
"""

import torch
from allennlp.nn import util


def batched_prune(items: torch.Tensor, features: torch.Tensor,
                  mask: torch.BoolTensor, scorer: torch.nn.Module, topk: int):
    """
    Prune based on mention scores.
    """

    # Shape: (batch_size, num_items)
    mention_scores = scorer(features).squeeze(-1)
    # Shape: (batch_size, topk) for all 3 tensors
    top_scores, top_mask, top_indices = util.masked_topk(mention_scores, mask, topk)

    # Shape: (batch_size * topk)
    # torch.index_select only accepts 1D indices, but here we need to select
    # items for each element in the batch. This reformats the indices to take
    # into account their index into the batch. We precompute this here to make
    # the multiple calls to util.batched_index_select below more efficient.
    flat_top_indices = util.flatten_and_batch_shift_indices(top_indices, items.size(1))

    # Compute final predictions for which items to consider as mentions.
    # Shape: (batch_size, topk, *)
    top_items = util.batched_index_select(items, top_indices, flat_top_indices)
    # Shape: (batch_size, topk, feature_size)
    top_features = util.batched_index_select(features, top_indices, flat_top_indices)

    return top_items, top_features, top_mask, top_indices, top_scores, flat_top_indices
