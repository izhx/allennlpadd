"""
PGN-Adapter-transformer in paper "Crowdsourcing Learning as Domain Adaptation",
https://arxiv.org/abs/2105.14980
"""

from typing import Optional, Dict, Any
from overrides.overrides import overrides

import torch
import torch.nn as nn

from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import TokenEmbedder

from .adapter_embedder import AdapterTransformerEmbedder


@TokenEmbedder.register("pgn_adapter_transformer")
class PgnAdapterTransformerEmbedder(AdapterTransformerEmbedder):
    """
    动态生成 adapter 参数
    """
    def __init__(
        self,
        model_name: str,
        *,
        domain_num: int,
        domain_embedding_dim: int = 8,
        pgn_layers: int = 12,
        share_param: bool = False,
        adapter_layers: int = 12,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        max_length: int = None,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name,
            adapter_layers=adapter_layers,
            adapter_kwargs=adapter_kwargs,
            external_param=True,
            max_length=max_length,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )

        if pgn_layers > adapter_layers:
            raise ConfigurationError(
                f"pgn_layers {pgn_layers} should less than adapter_layers {adapter_layers}")
        self.pgn_layers = pgn_layers
        self.share_param = share_param

        self.domain_embedding = nn.Embedding(domain_num, domain_embedding_dim)  # max_norm=1.0

        hidden_size = self.transformer_model.config.hidden_size
        adapter_size = adapter_kwargs["adapter_size"]
        size = [2] if share_param else [pgn_layers, 2]
        weights = dict(
            weight_down=nn.Parameter(torch.Tensor(
                *size, adapter_size, hidden_size, domain_embedding_dim)),
            weight_up=nn.Parameter(torch.Tensor(
                *size, hidden_size, adapter_size, domain_embedding_dim))
        )
        if self.adapters[0][0].bias:
            weights.update(
                bias_down=nn.Parameter(torch.zeros(
                    *size, adapter_size, domain_embedding_dim)),
                bias_up=nn.Parameter(torch.zeros(
                    *size, hidden_size, domain_embedding_dim))
            )
        self.weights = nn.ParameterDict(weights)

        self.preset_domain = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weights.weight_down, std=1e-3)
        nn.init.normal_(self.weights.weight_up, std=1e-3)

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
        domain: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if self.training or self.preset_domain is None:
            if domain is None:
                embedding = self.domain_embedding(self.preset_domain)
            else:
                embedding = self.domain_embedding(domain)
            self.generate_parameters(embedding)
        return super().forward(token_ids, mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask)

    def generate_parameters(self, embedding: torch.Tensor):
        def batch_matmul(w: torch.Tensor, e):
            ALPHA = "ijklmnopqrstuvwxyz"
            dims = ALPHA[:w.dim() - 1]
            i = 1 if self.share_param else 2
            return torch.einsum(f"{dims}a,ba->{dims[:i] + 'b' + dims[i:]}", w, e)

        matmul = batch_matmul if embedding.dim() == 2 else torch.matmul
        embedding = embedding.softmax(-1)
        weights = {k: matmul(v, embedding) for k, v in self.weights.items()}

        for i, adapters in enumerate(self.adapters[-self.pgn_layers:]):
            for j, adapter in enumerate(adapters):
                for k, v in weights.items():
                    param = v[j] if self.share_param else v[i, j]
                    setattr(adapter, k, param)
        return
