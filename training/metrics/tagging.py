from typing import Dict
from argparse import Namespace

import torch

from allennlp.training.metrics import Metric


class TaggingMetric(Metric):
    def __init__(self, ignore_index: int = 0):
        self.ignore_index = ignore_index
        self.counter = self.counter_factory()

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor,
                 mask: torch.LongTensor):
        mask = (gold_labels != self.ignore_index).long() * mask  # 只看标注
        self.counter.total += mask.sum().item()
        self.counter.positive += (
            (predictions != self.ignore_index).long() * mask).sum().item()
        self.counter.correct += (
            (predictions == gold_labels).long() * mask).sum().item()

    @staticmethod
    def counter_factory(total=0, positive=0, correct=.0) -> Namespace:
        # total: true num, positive: pred num, correct: tp num
        return Namespace(total=total, positive=positive, correct=correct)

    def compute_metric(self, counter: Namespace = None) -> Dict[str, float]:
        c = counter or self.counter
        total, correct, positive = c.total, c.correct, c.positive
        recall = 0 if total == 0 else correct / total
        precision = 0 if positive == 0 else correct / positive
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return dict(F1=f1, recall=recall, precision=precision)

    def reset(self) -> None:
        self.counter = self.counter_factory()

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metric = self.compute_metric()
        if reset:
            self.reset()
        return metric
