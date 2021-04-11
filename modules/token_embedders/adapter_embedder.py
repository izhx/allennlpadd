"""
Adapter for transformers in AllenNLP.

Parameter-Efficient Transfer Learning for NLPParameter-Efficient Transfer Learning for NLP
https://arxiv.org/abs/1902.00751
https://github.com/google-research/adapter-bert
"""

from typing import Optional, Dict, Any, Union, List

import torch
import torch.nn as nn
from torch.nn.functional import linear

from transformers import BertModel

from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders import (
    TokenEmbedder, PretrainedTransformerEmbedder)


@TokenEmbedder.register("adapter_transformer")
class AdapterTransformerEmbedder(PretrainedTransformerEmbedder):
    """
    目前只针对bert结构，插入adapter.
    """
    def __init__(
        self,
        model_name: str,
        *,
        adapter_size: int = 64,
        adapter_num: int = 12,
        external_param: Union[bool, List[bool]] = False,
        max_length: int = None,
        last_layer_only: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_name,
            max_length=max_length,
            train_parameters=False,
            last_layer_only=last_layer_only,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs
        )

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(adapter_num)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(adapter_num)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e
        else:
            raise ConfigurationError("wrong type of external_param!")

        self.adapters = nn.ModuleList([
            nn.ModuleList([
                Adapter(self.config.hidden_size, adapter_size, param_place[i]),
                Adapter(self.config.hidden_size, adapter_size, param_place[i])
            ]) for i in range(adapter_num)
        ])
        insert_adapters(self.transformer_model, self.adapters)


class Adapter(nn.Module):
    """
    Adapter module.
    """
    def __init__(self, in_features, bottleneck_size, external_param=False):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_size = bottleneck_size
        self.act_fn = nn.GELU()

        if external_param:
            self.w_down, self.b_down, self.w_up, self.b_up = None, None, None, None
        else:
            self.w_down = nn.Parameter(torch.Tensor(bottleneck_size, in_features))
            self.b_down = nn.Parameter(torch.zeros(bottleneck_size))
            self.w_up = nn.Parameter(torch.Tensor(in_features, bottleneck_size))
            self.b_up = nn.Parameter(torch.zeros(in_features))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.w_down, std=1e-3)
        nn.init.normal_(self.w_up, std=1e-3)

    def forward(self, hidden_states: torch.Tensor):
        linear_forward = batch_linear if self.w_down.dim() == 3 else linear
        x = linear_forward(hidden_states, self.w_down, self.b_down)
        x = self.act_fn(x)
        x = linear_forward(x, self.w_up, self.b_up)
        x = x + hidden_states
        return x


class AdapterBertOutput(nn.Module):
    """
    替代BertOutput和BertSelfOutput
    """
    def __init__(self, base, adapter_forward):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def set_requires_grad(module: nn.Module, status: bool = False):
    for param in module.parameters():
        param.requires_grad = status


def batch_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ batched linear forward """
    y = torch.einsum("bth,boh->bto", x, w)
    y = y + b.unsqueeze(1)
    return y


def insert_adapters(model, adapters_groups):
    if not isinstance(model, BertModel):
        raise ConfigurationError("目前只支持bert结构")

    for i, adapters in enumerate(adapters_groups, 1):
        layer = model.encoder.layer[-i]
        layer.output = AdapterBertOutput(layer.output, adapters[0].forward)
        set_requires_grad(layer.output.base.LayerNorm, True)
        layer.attention.output = AdapterBertOutput(layer.attention.output, adapters[1].forward)
        set_requires_grad(layer.attention.output.base.LayerNorm, True)

    return
